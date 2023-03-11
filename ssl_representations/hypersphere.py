import torch
from torch import nn
import torch.nn.functional as F

from utils import off_diagonal, FullGatherLayer
from projector import Projector

import timm


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
	z1: torch.Tensor
		Batch of intermediate representations which have been randomly augmented at the input level.

	z2: torch.Tensor
		Second batch of intermediate representations which have been randomly augmented at the input level. 

	Returns
	-------
	loss: torch.Tensor
		scalar loss value. 
	"""
	def __init__(self, dimension: int, batch_size: int, device: str):
		super().__init__()

		self.dimension = dimension
		self.batch_size = batch_size
		self.device
		self.sampler = vonMisesFisher(dimension = self.dimension, batch_size = self.batch_size, device = self.device)

	def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

		# draw vMF samples for z1 and z2, and compute SimCLR loss. 
		# break off mean and k from z1 and z2.
		pass

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
	Input: dimension m, mean \mu, concentration k
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
	Input: dimension m, concentration k
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
	Input: mean \mu, modal vector e_1
	u' = e_1 - \mu
	u = u' / ||u'||_2
	U =  I - 2 uu^T
	Return U
	"""

	arg_constraints = {

		"mean_direction": torch.distributions.constraints.real,
		"concentration": torch.distributions.constraints.positive

	}

	support = torch.distributions.constraints.real
	has_rsample = True 

	def __init__(self, mean_direction: torch.Tensor, concentration: torch.Tensor):
		super().__init__()

		#distribution params
		self.mean_direction = mean_direction
		self.concentration = concentration

		#tensor settings
		self.dtype = mean_direction.dtype
		self.device = mean_direction.device

		#dim settings
		self.dimension = mean_direction.shape[-1] #batch_size, dim
		self.batch_size = mean_direction.shape[0]

		#algorithm tmp tensors
		self.mode = torch.zeros(self.dimension, device=self.device)
		self.mode[0] = 1.0

	#the rsample method is needed for pathwise derivative
	def rsample(self) -> torch.Tensor:
		v = self._sample_v()
		w = self._accept_reject_w(k = k)
		# will batch_size, 1 * batch_size, dimension correctly broadcast?
		z_prime = torch.cat([w, torch.sqrt(1 - w.pow(2))*mean], dim=1)
		U = self._householder(mean)
		return U @ z_prime

	def _accept_reject_w(self) -> torch.Tensor:
		# k will be batch_size, 1

		b = (-2 * k + torch.sqrt(4*k + (self.dimension - 1)**2))/2. # batch_size, 1
		a = (self.dimension - 1) + 2*k + torch.sqrt(4*k**2 + (self.dimension-1)**2)/4.
		d = 4*a*b/(1 + b) - (self.dimension - 1)*torch.log(self.dimension-1)

		# may be a way to do this better, check official repo
		# I recall there was something with rolling the samples
		while True:
			eps = self._sample_beta(self.dimension)
			w = (1 - (1 + b)*eps)/(1 - (1 - b)*eps)
			t = 2*a*b / (1 - (1 - b)*eps)
			u = self._sample_uniform()

			lhs = (m - 1)*torch.log(t) - t + d 
			rhs = torch.log(u)
			if lhs >= rhs: 
				break
		return w # batch_size, 1

	def _householder(v: torch.Tensor) -> torch.Tensor:

		u_prime = self.mode - self.mean_direction
		u = u_prime / torch.linalg.norm(u_prime, dim=0)
		U = torch.eye(self.batch_size, device = self.device) - 2 * torch.outer(u, u)
		return U @ v

	def _sample_beta(dimension: int) -> float:
		# Sample from a Beta(1/2(m-1), 1/2(m-1)) distribution
		return eps

	def _sample_uniform():
		#sample from a uniform distribution U[0,1]

		return u

	def _sample_v():
		#sample a point uniformly on the unit sphere.
		v = torch.distributions.Normal(
			torch.Tensor(0, dtype=self.dtype, device=self.device), 
			torch.Tensor(1, dtype=self.dtype, device=self.device)
			).sample() #batch_size, 


		return v