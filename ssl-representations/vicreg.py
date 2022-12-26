import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss 
from torch.nn.functional import relu


class VarianceLoss(_Loss):
	r"""The variance loss, or the 'V' in VIC. This is a subclass of the internal loss class used
	for the pytorch non-weighted losses, _Loss. It is intialized in the same way that the internal
	non-weighted losses are, to be consistent with their internal losses. 

	The forward method of the class computes the standard deviation of the two representations
	over the batch dimension, and uses this in the hinge loss to encourage the standard deviations
	to be close to gamma, which is set to one. The mean is then taken over the encoding dimensions. 

	Let Z = [z_1, ..., z_n] be a batch of n representations, each of which is in R^d. 
	Let z^i denote the encoding dimensions, where i \in {1, ..., d}.

	The variance loss is computed as:

	v(Z) = \frac{1}{d} \sum_{j=1}^d \max(0, \gamma - S(z^j, \epsilon))

	where S(x, \epsilon) = \sqrt(\Var(x) + \epsilon), where the variance is taken over the batch. 
	"""
	def __init__(self, size_average = None, reduce = None, reduction: str = 'mean', gamma: float = 1.0, eps: float = 1e-4) -> None:
		super(VarianceLoss, self).__init__(size_average, reduce, reduction)
		self.gamma = gamma
		self.eps = eps

	def forward(self, z_a: Tensor, z_b: Tensor) -> Tensor:
		std_z_a = torch.sqrt(z_a.var(dim = 0) + self.eps) #compute variance over batch dimension
		std_z_b = torch.sqrt(z_b.var(dim = 0) + self.eps)

		return torch.mean(relu(self.gamma - std_z_a)) + torch.mean(relu(self.gamma - std_z_b))

class CovarianceLoss(_Loss):
	r"""The covariance loss, or the 'C'in VIC. This is a subclass of the internal loss class used 
	for the pytorch non-weighted losses, _Loss. It is intialized in the same way that the internal
	non-weighted losses are, to be consistent with their internal losses. 

	The forward method of the class computes the squared sum of the off-diagonal elements
	of the feature covariance matrix of the representation. First, the representation is centered
	using the mean over the batch. The feature covariance matrix is computed using the outer product
	of the deviation.

	Let Z = [z_1, ..., z_n] by a batch of n representations, each of which is in R^d.

	The feature covariance is computed as:

	Cov(Z) = \frac{1}{n} \sum_{i = 1}^n (z_i - \bar{z})(z_i - \bar{z})^T,   where \bar{z} = \frac{1}{n} \sum_{i = 1}^n z_i

	The loss is then computed as:

	c(Z) = \frac{1}{d} \sum_{i \neq j}[Cov(Z)]^2_{i, j}
	"""
	def __init__(self, size_average = None, reduce = None, reduction: str = 'mean') -> None:
		super(CovarianceLoss, self).__init__(size_average, reduce, reduction)

	def forward(self, z_a: Tensor, z_b: Tensor) -> Tensor:
		batch_size, D = z_a.shape

		z_a = z_a - z_a.mean(dim = 0)
		z_b = z_b - z_b.mean(dim = 0)

		cov_z_a = (z_a.t() @ z_a) / (batch_size - 1) #compute feature covariance over batch
		cov_z_b = (z_b.t() @ z_b) / (batch_size - 1)

		return off_diagonal(cov_z_a).pow_(2).sum() / D + off_diagonal(cov_z_b).pow_(2).sum() / D		 

def off_diagonal(M):
	r"""Clones a tensor, zero's it diagonal in-place, and returns the clone.

	Parameters
	----------

	M: torch.Tensor
		A square matrix for which the off-diagonals will be returned.

	O: torch.Tensor
		A square matrix in which the main diagonal has been zero'd.
	"""
	O = M.clone() 
	O.diagonal(dim1 = -1, dim2 = -2).zero_() #get diagonal and set to zero
	return O

