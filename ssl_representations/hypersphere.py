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

	In MCInfoNCE, for a given input x, a latent vector z is drawn from Unif(z; S^{D-1}), where D is the latent dimension.
	a positive view is drawn from vMF(z+; z, k_pos), and the negative views are drawn from Unif(z-, S^{D-1}).


	"""
	def __init__(self):
		super().__init__()

	def forward(self):
		pass