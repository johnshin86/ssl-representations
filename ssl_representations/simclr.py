import torch
from torch import nn
import torch.nn.functional as F

from utils import off_diagonal, FullGatherLayer
from projector import Projector

import timm

# add "boltzmann style" forward pass

class SimCLR(nn.Module):
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
		self.projector = Projector(args, self.rep_dim)
		self.criterion = nn.CrossEntropyLoss()
		
		self.tau = self.args.tau
		self.boltzmann = self.args.boltzmann
		self.sim_matrix_n = args.n_views * args.batch_size

		assert self.args.n_views == 2, "Currently, only 2 views are supported for InfoNCE loss."



	def forward(self, y1, y2):

		z1 = self.projector(self.backbone(y1))
		z2 = self.projector(self.backbone(y2))

		#Collect reps from all GPUs
		z1 = torch.cat(FullGatherLayer.apply(z1), dim=0)
		z2 = torch.cat(FullGatherLayer.apply(z2), dim=0)

		if self.tau:
			z1 = z1[:,:-1]
			z2 = z2[:,:-1]

			tau1 = torch.sigmoid(z1[:,-1].unsqueeze(1)) # batch_size, 1
			tau2 = torch.sigmoid(z2[:,-1].unsqueeze(1)) # batch_size, 1 
			
			tau = torch.concat([tau1, tau2], dim=0) # 2 * batch_size, 1
			tau = tau.repeat(1, self.args.n_views * self.args.batch_size) # 2 * batch_size, 2 * batch_size

			# This should be a block matrix of
			# | t1 | t1 |
			# | t2 | t2 |

		# z1, z2 should be args.batch_size == per_device_batch_size * args.world_size
		# i.e. args.batch_size is the total across GPUs

		z = torch.concat([z1, z2], dim=0)

		if not self.boltzmann:

			# create "true" similarity matrix
			true_sim = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
			true_sim = (true_sim.unsqueeze(0) == true_sim.unsqueeze(1)).float()
			true_sim = true_sim.to(self.args.device)

			#normalize features
			z = F.normalize(z, dim=1)

			# construct similarity matrix
			similarity_matrix = torch.matmul(z, z.T)

			if self.tau:
				#weight it with inverse temperature. 
				similarity_matrix = similarity_matrix * tau

			# delete diagonals and shift left
			mask = torch.eye(true_sim.shape[0], dtype=torch.bool).to(self.args.device)
			true_sim = true_sim[~mask].view(true_sim.shape[0], -1)
			similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

			# each index is given it's positive similarity score
			positives = similarity_matrix[true_sim.bool()].view(true_sim.shape[0], -1)

			# each index is given all 2 * (N - 2) dissimilarity scores
			negatives = similarity_matrix[~true_sim.bool()].view(similarity_matrix.shape[0], -1)

			# concat positive examples to index 0
			# make index 0 the label for each sample
			logits = torch.cat([positives, negatives], dim=1)
			labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
			logits = logits / self.args.temp

			# At first glance, this does not look like the SimCLR loss. 
			# Using cross-entropy loss in this manner will consider the 0 index the positive view,
			# and the other n_view * (N - n_view) indices the negative views. 
			# Hence we have -log [ exp(s_{i,j} / T) / ( \sum_{k != i} exp(s_{i, k}) ) ] as well
			# as its symmetric counterpart j, i, for all i in [0, N]. 
			loss = self.criterion(logits, labels)

		else:

			similarity_matrix = torch.matmul(z, z.T)

			if self.tau:
				similarity_matrix = (similarity_matrix * tau) 

			# create matrix where off-diagonals are all true
			mask = ~torch.eye(self.sim_matrix_n, device=self.args.device).bool()
			# select off-diagonals, reshape
			neg = similarity_matrix.masked_select(mask).view(self.sim_matrix_n, -1)

			pos = torch.sum(z1 * z2, dim=-1)

			if self.tau:
				pos = pos * (tau1).squeeze(-1)

			pos = torch.cat([pos, pos], dim=0)
			logits = torch.cat([pos.unsqueeze(-1), neg], dim=1)
			labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
			logits = logits / self.args.temp

			loss = self.criterion(logits, labels)

		return loss
