r"""
Inverting a representation to find the image corresponding to the representation
requires some subtelty in the reconstruction objective. The naive attempt at optimizing
the l2 reconstruction error of the representation ||r_0 - r(x)||^2_2 suffers 
from extreme noise and high frequency effects. Gradient descent to find a solution in this
manner often recovers something akin to an adversarial example. Additional regularization 
must be imposed to recover solutions with support over the natural images.
There are a taxonomy of such regularizers, covered in:

https://distill.pub/2017/feature-visualization/#enemy-of-feature-vis

One simple regularizer is covered in the work of:

Mahendran, Aravindh, and Andrea Vedaldi. "Understanding deep image representations by inverting them." 
Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

The objective is given by:

||r(s * x) - r_0||^2_2 + \lambda R_{\alpha}(x) + \lambda_{V^{\beta}} R_{V^{\beta}} (x)


Where R_{\alpha} is the alpha norm, R_{V^{\beta}} is the total variation, and s is the average
euclidean norm of the training images.

The two lambda coefficients are chosen based on the heuristics:

\lambda_{\alpha} \approx \sigma^{\alpha}/(HWB^{\alpha})
\lambda_{\beta} \approx \sigma^{\alpha}/(HW(aB)^{\alpha})

Where a \approx .01 is a coefficient relating the dynamic range of the image to its gradient.
"""
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss 

class TotalVariationLoss(_Loss):
	r"""
	"""
	def __init__(self):
		super().__init__()

	def forward(self, img: Tensor, beta: float) -> Tensor:
		tv_h = img - img
		tv_w = img - img
		tv_h[:,:,1:,:] = (img[:,:,1:,:] - img[:,:,:-1,:]).pow(2)
		tv_w[:,:,:,1:] = (img[:,:,:,1:] - img[:,:,:,:-1]).pow(2)
		return (tv_h + tv_w).pow(beta/2.).sum()