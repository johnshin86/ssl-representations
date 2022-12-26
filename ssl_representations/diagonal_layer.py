import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
import math


#TODO: implement as CUDA kernel.

class DiagonalLinear(torch.nn.Module):
	r"""Implementation of a diagonal weight matrix.

	The regular Linear module in pytorch applies the linear transformation y = xA^T + b
	to the incoming data. 

	This DiagonalLinear module assumes that A is diagonal, with min(in_features, out_features)
	as the degrees of freedom. 
	"""
	__constants__ = ['in_features', 'out_features']
	in_features: int
	out_features: int 
	weight: Tensor

	def __init__(self, in_features: int, out_features: int, bias: bool = True, device = None, dtype = None) -> None:
		factory_kwargs = {'device': device, 'dtype': dtype}

		super(DiagonalLinear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.min = min([in_features, out_features])
		self.diag = Parameter(torch.eye(n = self.in_features, m = self.out_features, **factory_kwargs), requires_grad=False)
		self.weight = Parameter(torch.empty(self.min, **factory_kwargs))
		if bias:
			self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self) -> None:
		# Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
		# uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
		# https://github.com/pytorch/pytorch/issues/57109
		bound = 1/math.sqrt(self.min)
		init.uniform_(self.weight, -bound, bound)
		if self.bias is not None:
			bound = 1/math.sqrt(len(self.bias))
			init.uniform_(self.bias, -bound, bound)



	def forward(self, input: Tensor) -> Tensor:

		# TODO: this implementation is pretty slow. 

		# A is output x input
		# A^T is input x output
		W = torch.diag(self.weight)
		
		if self.min == self.out_features:
			# in_features x out_features @ out_features x out_features = in_features x out_features

			if self.bias is not None:

				return input @ self.diag @ W  + self.bias
			
			else:

				return input @ self.diag @ W  
		else:
			# in_features x in_features @ in_features x out_features = in_features x out_features

			if self.bias is not None:

				return input @ W @ self.diag + self.bias

			else:

				return input @ W @ self.diag


	def extra_repr(self) -> str:
		return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)