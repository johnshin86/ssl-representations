import torch
from torch import nn
import torch.nn.functional as F
import math

from utils import off_diagonal, FullGatherLayer
from projector import Projector

import timm

from typing import Tuple

class MCInfoNCE(nn.Module):
	r"""
	In this method, we interpret the feature vector and (inverse) temperature of the SimCLR framework
	as the mean direction and concentration parameter of the von Mises-Fisher (vMF) distribution. 
	The distribution is of the form:

	f_p (\vz; \mu, k) = C_p(k)exp(k \mu^T \vz)

	Where k >= 0, ||\mu|| = 1, and C_p(k) is a normalization constant of the form:

	C_p(k) = \frac{k^{p/2 - 1}}{(2\pi)^{p/2}I_{p/2 - 1}(k)}

	where I_{v} denotes the modified Bessel function of the first kind at order v. 

	The SimCLR method essentially views the two views of a sample as positively correlated,
	and the other samples in the batch as negatively correlated. The measure of correlation is
	the cosine between the two vectors. 

	We assume that there is a natural generative process g that transforms latent components z \in Z into
	the images x = g(z), x \in X. We assume Z to be S^{D-1}, or the D dimensional hypersphere. We are interested
	in recovering the latents z for images x. We assume g: Z -> X is an injective and deterministic function.
	Thus only one z can correspond to each x, and g is invertible. An encoder f trained with a contrastive
	InfoNCE loss achieves this inversion and recovers the correct latent z:

	f(x) = f(g(z)) = \hat{z} = Rz, up to an orthogonal rotation R. 

	However, the image x may be corrupted, blurred, low resolution, or otherwise underdetermined. 
	In this case, the generative process g is non-injective and our encoder can only recover a set
	of possible latents. g may also be stochastic. 

	We can model g as the likelihood P(x|z). Instead of explicitly characterizing g by P(x|z)
	we implicitly characterize it by its posterior P(z|x). 

	We parameterize P(z|x) by a vMF distribution:

	P(z|x) = C(k(x))exp(k(x)\mu(x)^Tz)

	Suppose we have (x, x+, x-_1, ..., x-_M), or a reference sample x, a positive sample x^+,
	and negative samples x-_1, ... , x-_M. We assume that these samples are generated from the latents
	z, z+, z-_1, ... , z-_M. The latents are drawn:

	z ~ P(z) = Unif(z; S^{D-1})
	z+ ~ P(z+|z) = vMF(z+; z, k_pos)
	z- ~ P(z-|z) = P(z-) = Unif(z-; S^{D-1})

	The fixed constant k_pos controls how close latents must be to be considered positive (and is different than the concentration
	parameter k(x)). The latents are transformed into observations via the generative process P(x|z). This defines P(x), P(x+|x) and P(x-).

	The encoder f outputs probabilistic embeddings Q(z|x) = vMF(z; \hat{mu}(x), \hat{k}(x)) by predicting f(x) = (\hat{\mu}(x), \hat{k}(x)).

	The loss function is given as:

	L := E[L_f(x, x+, x-_1, ..., x-_M)]
		x ~ P(x)
		x+ ~ P(x+|x)
		x-_m ~ P(x-)

	L_f := -log E[exp(k_pos z^Tz+) / {1/M exp(k_pos z^T z+) + 1/M \sum_{m=1}^M exp(k_pos z^T z-_m)}]
		z ~ Q(z|x)
		z+ ~ Q(z+|x+)
		z-_m ~ Q(z-_m | x-_m)

	It can be proved that the optimizer for this loss learns the correct location \hat{\mu}(x) = R \mu(x), up to a constant orthogonal rotation R,
	and the correct level of ambiguity \hat{k}(x) = k(x) for each observation x.

	Parameters
	----------
	y1: torch.Tensor
		Batch of intermediate representations which have been randomly augmented at the input level.

	y2: torch.Tensor
		Second batch of intermediate representations which have been randomly augmented at the input level. 

	Returns
	-------
	loss: torch.Tensor
		scalar loss value. 
	"""
	def __init__(self, args, n_samples: int = 20, k: int = 20):
		super().__init__()
		self.args = args
		
		#init model
		if "resnet" in args.arch:
			model = timm.create_model(args.arch, zero_init_last=True)
		else:
			model = timm.create_model(args.arch)

		#backbone and projector settings
		self.rep_dim = model.fc.in_features
		model.fc = nn.Identity()
		self.backbone = model
		self.projector = Projector(args, self.rep_dim)
		self.criterion = nn.CrossEntropyLoss()

		#loss settings
		self.sim_matrix_n = args.n_views * args.batch_size
		self.n_samples = n_samples
		self.k = k
		self.etol = 1e-4
		self.relu = torch.nn.ReLU()

		assert self.args.n_views == 2, "Currently, only 2 views are supported for MCInfoNCE loss."

	def forward(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:

		u1 = self.projector(self.backbone(y1))
		u2 = self.projector(self.backbone(y2))

		#Collect reps from all GPUs
		u1 = torch.cat(FullGatherLayer.apply(u1), dim=0)
		u2 = torch.cat(FullGatherLayer.apply(u2), dim=0)

		u1 = u1[:,:-1]
		u2 = u2[:,:-1]

		kappa1 = self.relu(u1[:,-1]) + self.etol #  batch_size
		kappa2 = self.relu(u2[:,-1]) + self.etol # batch_size
		
		kappa = torch.concat([kappa1, kappa2], dim=0) # 2 * batch_size

		# u1, u2 should be args.batch_size == per_device_batch_size * args.world_size
		# i.e. args.batch_size is the total across GPUs

		u1 = F.normalize(u1, dim=1)
		u2 = F.normalize(u2, dim=1)
		u = torch.concat([u1, u2], dim=0)

		vMF = vonMisesFisher(loc = u, scale = kappa, k = 20)
		z = vMF.rsample(shape=torch.Size([self.n_samples])) # n_samples, 2 * batch_size, embedding_dim
		assert len(z.shape) == 3, "The length of the shape of z here should be 3."
		z = torch.mean(z, dim=0) # sum over MC samples
		
		similarity_matrix = torch.matmul(z, z.T)

		# create matrix where off-diagonals are all true
		mask = ~torch.eye(self.sim_matrix_n, device=self.args.device).bool()
		# select off-diagonals, reshape
		neg = similarity_matrix.masked_select(mask).view(self.sim_matrix_n, -1)

		#this can be done cleaner
		pos = torch.sum(z[:self.args.batch_size, ] * z[self.args.batch_size:, ], dim=-1)
		pos = torch.cat([pos, pos], dim=0)
		# pos = similarity_matrix[:self.args.batch_size, self.args.batch_size:].diag()
		# pos = torch.cat([pos, pos], dim = 0)

		logits = torch.cat([pos.unsqueeze(-1), neg], dim=1)
		labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
		logits = logits / self.args.temp

		loss = self.criterion(logits, labels)

		return loss


class vonMisesFisher(torch.distributions.Distribution):
	r"""Allows for the sampling from a von Mises Fisher distribution.
	The torch.distributions.Distribution class allows for the backpropagation
	through this sampling. If the PDF is differentiable with respect to its
	parameters, the sample() and log_prob() methods need to be implemented
	to use REINFORCE. Otherwise, one can use the pathwise derivative,
	and the rsample() method must be implemented.

	The algorithm is as follows:
	
	Algorithm 1
	-----------
	Input: dimension m, loc \mu, scale k
	sample v ~ U(S^{m-2})
	sample w ~ g(w|k, m) \propto exp(wk)(1 - w^2)^{(m-3)/2}
	accept-reject procedure (Algorithm 2)
	z' = (w; sqrt(1-w^2)v^T)^T
	U = Householder(e_1, \mu) (Algorithm 3)
	Return Uz'

	We sample from a vMF q(z|e_1, k), with e_1 as the mean direction.
	The vMF density is uniform in all the m-2 dimensional sub-hyperspheres. 
	{x \in S^{m-1}|e^T_1 \vx = w}, and hence the sampling technique reduces
	to sampling the scalar value w from the univariate density

	g(w|k, m) \propto  exp(kw)(1-w^2)^{(m-3)/2}, w \in [-1, 1]

	using an accept-reject scheme. 

	After getting a sample from q(z|e_1, k) an orthogonal transformation
	U(\mu) is applied such that the transformed sample is distributed
	according to q(z|\mu, k). This can done with a Householder
	reflection U(\mu)e_1 = \mu.

	The accept-reject procedure is as follows:

	Algorithm 2
	-----------
	Input: dimension m, scale k
	Initialize values:
	b = (-2k + \sqrt{4k^2 + (m-1)^2})/2
	a = ((m-1) + 2k + \sqrt{4k^2 + (m-1)^2})/4
	d = 4ab/(1 + b) - (m-1)ln(m-1)

	repeat
		Sample \eps ~ Beta(1/2(m-1), 1/2(m-1))
		w = h(\eps, k) = (1 - (1 + b)\eps)/(1 - (1 - b)\eps)
		t = 2ab / (1 - (1 - b)\eps)
		Sample u ~ U(0,1)
	until (m-1)ln(t) - t + d >= ln(u)
	Return w

	Finally, we have the algorithm for the Householder transformation.

	Algorithm 3
	-----------
	Input: loc \mu, modal vector e_1
	u' = e_1 - \mu
	u = u' / ||u'||_2
	U =  I - 2 uu^T
	Return U
	"""

	arg_constraints = {

		"loc": torch.distributions.constraints.real,
		"scale": torch.distributions.constraints.positive

	}

	support = torch.distributions.constraints.real
	has_rsample = True 

	def __init__(self, loc: torch.Tensor, scale: torch.Tensor, validate_args = None, k: int = 1):

		#distribution params
		self.loc = loc
		self.scale = scale

		#tensor settings
		self.dtype = loc.dtype
		self.device = loc.device

		#dim settings
		self.dim = loc.shape[-1] #batch_size, dim
		self.batch_size = loc.shape[0]

		#algorithm settings
		self.e1 = torch.zeros(self.dim, device=self.device)
		self.e1[0] = 1.0
		self.etol = 1e-14
		self.k = k

		super().__init__(self.loc.size(), validate_args=validate_args)

	#the sample method is used for REINFORCE
	def sample(self, shape = torch.Size()):
		with torch.no_grad():
			return self.rsample(shape)

	#the rsample method is needed for pathwise derivative
	def rsample(self, shape = torch.Size()) -> torch.Tensor:
		shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])

		w = self._sample_w(shape = shape)
		v = self._sample_v(shape = shape)

		w_cos = torch.sqrt(torch.clamp(1 - w**2, self.etol))
		x = torch.cat((w.unsqueeze(-1), w_cos.unsqueeze(-1)*v), -1)
		z = self._householder(x)
		z = z.type(self.dtype)

		if torch.any(torch.isnan(z)):
			return self.rsample(shape)
		else:
			return z

	def _sample_w(self, shape) -> torch.Tensor:
		quad = torch.sqrt( (4 * self.scale**2) + (self.dim - 1)**2)
		
		#the true b
		b_true = (-2 * self.scale + quad) / (self.dim - 1)

		# k = inf expansion of b(k)
		b_approx = (self.dim - 1) / (4 * self.scale)

		# this is a "linear" heaviside which linearly transitions from 0 to 1 in between 10 to 11
		mix = torch.min(
				torch.max(torch.tensor([0.0], dtype=self.dtype, device=self.device), self.scale - 10
					),
				torch.tensor([1.0], dtype=self.dtype, device=self.device)
			)

		# when k > 11, use the k=inf expansion. when k < 10 is small, use the true b.
		# when 10 < k < 11, use a linear mix.
		b = b_approx * mix + b_true * (1 - mix)

		a = (self.dim - 1 + 2 * self.scale + quad) / 4
		d = (4 * a * b) / (1 + b) - (self.dim - 1) * math.log(self.dim - 1)

		self.b = b
		self.eps, self.w = self._while_w(b, a, d, shape, self.k)

		return self.w

	@staticmethod
	def first_nonzero(x, dim, invalid_val = -1):
		# check operations with dummy tensors
		# mask nonpositive entries
		mask = x > 0
		idx = torch.where(
			# check if any entry is true
			mask.any(dim=dim),
			# return index of max, max will be first non-zero since all non-zeros are ones.
			mask.float().max(dim=1)[1].squeeze(),
			#otherwise return the index for having not found it, which is -1
			torch.tensor(invalid_val, device=x.device)

			)

		return idx
		

	def _while_w(self, b, a, d, shape, k=20) -> Tuple[torch.Tensor, torch.Tensor]:
		is_inf = self.scale == float("inf")

		# Instead of sampling (# of samples), we sample (# of samples) * k 
		# Having k for the i'th sample allows us to have additional samples in case of failure

		# for each constant, repeat it's value (# of samples) * conc.shape 

		# b.shape is the same shape as conc (batch_size, 1)
		b, a, d, is_inf = [
			e.repeat(*shape, *([1] * len(self.scale.shape))).reshape(-1,1)
			for e in (b, a, d, is_inf)
		]

		# Init tensors for w, e, and a mask in the same shape
		w, e, bool_mask = (
			torch.zeros_like(b, device=self.device),
			torch.zeros_like(b, device=self.device),
			(torch.ones_like(b, device=self.device) == 1)
		)

		# sample_shape is (# of samples), k
		# shape is (# of samples), conc.shape 
		sample_shape = torch.Size([b.shape[0], k])
		shape = shape + torch.Size(self.scale.shape)

		# while any bool_mask entry is nonzero
		while bool_mask.sum() != 0:

			#sample eps and u (# of samples, k)
			eps = self._sample_beta(sample_shape)
			u = self._sample_uniform(sample_shape)

			#compute w_tmp and t for comparison, we will only copy over to w if it's an accept
			#they will be of size (# of samples, k)
			w_tmp = (1 - (1 + b) * eps) / (1 - (1 - b) * eps)
			t = (2 * a * b) / (1 - (1 - b) * eps)

			accept = ((self.dim - 1.0) * t.log() - t + d) > torch.log(u)

			#return mean value of w (w=1) if k=inf infinite
			#w_tmp is actually larger than b due to the k factor
			w_tmp[is_inf.squeeze(),] = torch.ones(k, device=self.device)
			accept[is_inf.squeeze(),] = torch.ones(k, device=self.device).bool()

			#pick the first accept index along the k dimension, clamp invalids
			accept_idx = self.first_nonzero(accept, dim=-1, invalid_val=-1).unsqueeze(1)
			accept_idx_clamped = accept_idx.clamp(0)

			#gather along axis dim=1 (k-axis) where accept indices were found
			w_tmp = w_tmp.gather(1, accept_idx_clamped.view(-1,1))
			eps = eps.gather(1, accept_idx_clamped.view(-1,1))

			# reject mask where one finds invalid_index = -1
			# accept mask is negation of that
			reject = accept_idx < 0
			accept = ~reject

			#copy over the accept entries to w and e over the remaining bool_mask
			w[bool_mask * accept] = w_tmp[bool_mask * accept]
			e[bool_mask * accept] = eps[bool_mask * accept]

			# turn off bool mask along accepted indices
			bool_mask[bool_mask * accept] = reject[bool_mask * accept]

			#continue until bool mask is all zeros
			#one inefficiency with this is that it samples (# of samples), k every time. 

			#TODO: refactor so sample_shape is not sampled when only the remaining bool_mask needs to be filled. 

		return e.reshape(shape), w.reshape(shape)


	def _householder(self, z: torch.Tensor) -> torch.Tensor:
		# performs the householder transform on z
		# shape tested
		u = self.e1 - self.loc
		u_prime = u / u.norm(dim=-1, keepdim = True).clamp_min(self.etol)
		mu = z - 2 * (z * u_prime).sum(-1, keepdim=True) * u_prime 
		return mu
		
	def _sample_beta(self, shape) -> torch.Tensor:
		# Sample eps ~ Beta(1/2(m-1), 1/2(m-1))

		#decoupling constants
		alpha = torch.tensor((self.dim - 1)/2, dtype=torch.float64, device=self.device)
		beta = torch.tensor((self.dim - 1)/2, dtype=torch.float64, device=self.device)
		eps = torch.distributions.Beta(alpha, beta).sample(shape).type(self.dtype)
		
		return eps

	def _sample_uniform(self, shape) -> torch.Tensor:
		#sample u ~ U[0,1]
		#shape tested
		u = torch.distributions.Uniform(
			torch.tensor(0 + self.etol, dtype=self.dtype, device=self.device),
			torch.tensor(1 - self.etol, dtype=self.dtype, device=self.device),
			).sample(shape)

		return u

	def _sample_v(self, shape) -> torch.Tensor:
		#sample a point uniformly on the unit sphere S^{m - 2}.
		#shape tested
		v = torch.distributions.Normal(
			torch.tensor(0, dtype=self.dtype, device=self.device), 
			torch.tensor(1, dtype=self.dtype, device=self.device),
			).sample(
			#add n_samples to batch,dim
			shape + torch.Size(self.loc.shape)
			).transpose(0, -1)[1:].transpose(0,-1)
			 #batch_size,

		v = v/v.norm(dim=-1, keepdim=True)

		return v