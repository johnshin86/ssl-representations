import torch
from torch import nn
import torch.nn.functional as F

from utils import off_diagonal, FullGatherLayer
from projector import Projector

import timm

#from simclr import SimCLR

# Should we just add this to SimCLR?

class SimCLR_tau(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.embedding_dim = int(args.mlp.split("-")[-1])

		if "resnet" in args.arch:
			model = timm.create_model(args.arch, zero_init_last=True)
		else:
			model = timm.create_model(args.arch)

		self.rep_dim = model.fc.in_features
		model.fc = nn.Identity()
		self.backbone = model
		self.projector = Projector_tau(args, self.rep_dim)
		self.criterion = nn.CrossEntropyLoss()

		assert self.args.n_views == 2, "Currently, only 2 views are supported for InfoNCE loss."



	def forward(self, y1, y2):
		z1 = self.projector(self.backbone(y1))
		z2 = self.projector(self.backbone(y2))

		# break off tau here ... 

		#Collect reps from all GPUs
		z1 = torch.cat(FullGatherLayer.apply(z1), dim=0)
		z2 = torch.cat(FullGatherLayer.apply(z2), dim=0)

		# z1, z2 should be args.batch_size == per_device_batch_size * args.world_size
		# i.e. args.batch_size is the total across GPUs

		z = torch.concat([z1, z2], dim=0)

		# create "true" similarity matrix
		true_sim = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
		true_sim = (true_sim.unsqueeze(0) == true_sim.unsqueeze(1)).float()
		true_sim = true_sim.to(self.args.device)

		#normalize features
		z = F.normalize(z, dim=1)

		# construct similarity matrix
		similarity_matrix = torch.matmul(z, z.T)

		# delete diagonals and shift left
		mask = torch.eye(true_sim.shape[0], dtype=torch.bool).to(self.args.device)
		true_sim = true_sim[~mask].view(true_sim.shape[0], -1)
		similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

		# each index is given it's positive similarity score
		positives = similarity_matrix[true_sim.bool()].view(true_sim.shape[0], -1)

		# each index is given all 2 * (N - 2) dissimilarity scores
		negatives = similarity_matrix[~true_sim.bool()].view(similarity_matrix.shape[0], -1)

		# concat positive examples to index 0
		logits = torch.cat([positives, negatives], dim=1)
		# make index 0 the label for each sample
		labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
		logits = logits / self.args.temp

		# At first glance, this does not look like the SimCLR loss. 
		# Using cross-entropy loss in this manner will consider the 0 index the positive view,
		# and the other n_view * (N - n_view) indices the negative views. 
		# Hence we have -log [ exp(s_{i,j} / T) / ( \sum_{k != i} exp(s_{i, k}) ) ] as well
		# as its symmetric counterpart j, i, for all i in [0, N]. 
		loss = self.criterion(logits, labels)

		return loss

class SimCLR(object):

    def __init__(self, outputs1, outputs2, t=0.07, eps=1e-6):
        super().__init__()
        self.outputs1 = F.normalize(outputs1, dim=1)
        self.outputs2 = F.normalize(outputs2, dim=1)
        #t was likely found through linear search
        self.eps, self.t = eps, t

    def get_loss(self, split=False):
        # out: [2 * batch_size, dim]
        out = torch.cat([self.outputs1, self.outputs2], dim=0)
        n_samples = out.size(0)

        # cov and sim: [2 * batch_size, 2 * batch_size]
        # neg: [2 * batch_size]

        #compute gram matrix (not actually covariance!)
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / self.t)

        # This isn't fidelitous to the paper since it also contains the positive `view."
        # But it is more fidelitous to the boltzman factor analogy
        
        # Create boolean mask, true on off-diagonals
        mask = ~torch.eye(n_samples, device=sim.device).bool()
        # select all the negative samples and sum them
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(self.outputs1 * self.outputs2, dim=-1) / self.t)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + self.eps)).mean()

        if not split:
            return loss
        else:
            # return overall loss, numerator, denominator
            return loss, pos, (neg + self.eps)


class TaU_SimCLR(object):
    
    def __init__(self, loc1, temp1, loc2, temp2, t=0.07, eps=1e-6, simclr_mask=False):
        super().__init__()
        self.loc1 = F.normalize(loc1, dim=1)
        self.loc2 = F.normalize(loc2, dim=1)
        self.temp1 = torch.sigmoid(temp1)
        self.temp2 = torch.sigmoid(temp2)
        self.eps, self.t = eps, t
        self.simclr_mask = simclr_mask

    def build_mask(self, mb, device, simclr=False): # Either building the SimCLR mask or the new mask
        if simclr:
            m = torch.eye(mb, device=device).bool()
        else:
            m = torch.eye(mb // 2, device=device).bool()
            m = torch.cat([m, m], dim=1)
            m = torch.cat([m, m], dim=0)
        return m

    def get_loss(self, split=False):
        # out: [2 * batch_size, dim]
        loc = torch.cat([self.loc1, self.loc2], dim=0)
        temp = torch.cat([self.temp1, self.temp2], dim=0)
        n_samples = loc.size(0)

        # cov and sim: [2 * batch_size, 2 * batch_size]
        # neg: [2 * batch_size]
        cov = torch.mm(loc, loc.t().contiguous())

        # This is also incorrect, as it "interleaves" the temps instead of doing temp1*batch, temp2*batch 
        # (2 * n_samples)
        var = temp.repeat(1, n_samples)
        sim = torch.exp((cov * var) / self.t)

        # NOTE: this mask is a little different than SimCLR's mask
        mask = ~self.build_mask(n_samples, sim.device, simclr=self.simclr_mask)
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.sum(self.loc1 * self.loc2, dim=-1)
        pos = pos * (self.temp1 / self.t).squeeze(-1)
        pos = torch.exp(pos)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + self.eps)).mean()

        if not split:
            return loss
        else:
            # return overall loss, numerator, denominator
            return loss, pos, (neg + self.eps)
