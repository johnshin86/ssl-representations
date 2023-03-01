import torch
from torch import nn
import torch.nn.functional as F

from utils import off_diagonal, FullGatherLayer
from projector import Projector

import timm


class MCInfoNCE(nn.Module):
	r"""
	In this method, we interpret the feature vector and (inverse) temperature of the SimCLR framework
	as the mean direction and concentration parameter of the von Mises-Fisher (vMF) distribution. The distribution
	is of the form:

	f_p (\vz; \mu, k) = C_p(k)exp(k \mu^T \vz)

	Where k >= 0, ||\mu|| = 1, and C_p(k) is a normalization constant of the form:

	C_p(k) = \frac{k^{p/2 - 1}}{(2\pi)^{p/2}I_{p/2 - 1}(k)}

	where I_{v} denotes the modified Bessel function of the first kind at order v. 

	The SimCLR method essentially views the two views of a sample as positively correlated,
	and the other samples in the batch as negatively correlated. 

	We assume that there is a natural generative process g that transforms latent components z \in Z into
	the images x = g(z). We assume Z to be S^{D-1}, or the D dimensional hypersphere.

	We parameterize P(z|x) by a vMF distribution:

	P(z|x) = C(k(x))exp(k(x)\mu(x)^Tz)

	Suppose we have (x, x+, x-_1, ..., x-_M), or a reference sample x, a positive sample x^+,
	and negative samples x-_1, ... , x-_M. We assume that these samples are generated from the  latents
	z, z+, z-_1, ... , z-_M. The latent vector z is drawn from Unif(z; S^{D-1}), where D is the latent dimension.
	a positive view is drawn from vMF(z+; z, k_pos), and the negative views are drawn from Unif(z-, S^{D-1}).
	The fixed constant k_pos controls how close latents must be to be considered positive (and is different than k(x)). 
	The latents are transformed into observations via the generative process P(x|z). This defines P(x), P(x+|x) and P(x-).

	The encoder f outputs probabilistic embeddings Q(z|x) = vMF(z; \mu(x), k(x))


	"""
	def __init__(self):
		super().__init__()

	def forward(self):
		pass