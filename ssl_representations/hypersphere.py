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
	and negative samples x-_1, ... , x-_M. We assume that these samples are generated from the  latents
	z, z+, z-_1, ... , z-_M. The latens are drawn:

	z ~ P(z) = Unif(z; S^{D-1})
	z+ ~ P(z+|z) = vMF(z+; z, k_pos)
	z- ~ P(z-|z) = P(z-) = Unif(z-; S^{D-1})

	The fixed constant k_pos controls how close latents must be to be considered positive (and is different than k(x)). 
	The latents are transformed into observations via the generative process P(x|z). This defines P(x), P(x+|x) and P(x-).

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



	"""
	def __init__(self):
		super().__init__()

	def forward(self):
		pass

class vonMisesFisher(torch.distributions.Distribution):
	