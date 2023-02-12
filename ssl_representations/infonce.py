import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss 
from torch.nn.functional import relu


class InfoNCE(_Loss):
	r"""
	"""
	def __init__(self, size_average = None, reduce = None, reduction: str = 'mean') -> None:
		super(InfoNCE, self).__init__(size_average, reduce, reduction)

	def forward(self, z_a: Tensor, z_b: Tensor) -> Tensor:
		# Build similarity matrix

		z = torch.cat( (z_a, z_b), dim = 0) # 2N x F

		z_norm = torch.linalg.norm(z, ord = 2, dim = 1) # Norm over F

		z_normalized = z / z_norm[:, None]

		similarity = z_normalized @ z_normalized.T 

		#masking?

		return 


