import torch
from torch import nn
import torch.nn.functional as F

from utils import off_diagonal, FullGatherLayer
from projector import Projector

import timm


class MCInfoNCE(nn.Module):
	r"""
	In this method, we interpret the feature vector and (inverse) temperature of the SimCLR framework
	as the mean direction and concentration parameter of the von Mises-Fisher distribution. The distribution
	is of the form:

	f_p (\vx; \mu, k) = C_p(k)exp(k \mu^T \vx)

	Where k >= 0, ||\mu|| = 1, and C_p(k) is a normalization constant of the form:

	C_p(k) = \frac{k^{p/2 - 1}}{(2\pi)^{p/2}I_{p/2 - 1}(k)}

	where I_{v} denotes the modified Bessel function of the first kind at order v. 
	"""
	def __init__(self):
		super().__init__()

	def forward(self):
		pass