# SSL Representations

Self-Supervised Learning := SSL

This library implements many recent vision SSL methods, as well as uncertainty estimation methods built on top of these methods.
Rather than aiming to implement every SSL method, the goal of this library is to focus on understanding conceptually new methods
and their theory and rigorous application, as well as testing of the produced representation quality. 

Self-supervised learning in vision is an interplay of the augmentation policy, the inductive bias of the architecture,
and the interactions across samples induced by the loss function (https://arxiv.org/abs/2302.02774).  The optimal representations can be mapped to different spectral embedding methods (https://arxiv.org/abs/2205.11508), though this requires a "god's eye view" of the underlying similarity matrix. Contrastive and non-contrastive methods can be viewed through the lens
of Gram and covariance matrices, respectively (https://arxiv.org/abs/2206.02574).

The library was initially built from the facebook VICReg repo, but has been heavily modified.

# Instructions on Use

For training, run the command:

`python3 train.py --framework <vicreg/barlowtwins/simclr> --dataset <cifar10/imagenet> --data-dir <path/to/data> `

An inference/evaluation pipeline is not yet implemented. 

# Methods Implemented

- VICReg (https://arxiv.org/abs/2105.04906)
- Barlow Twins (https://arxiv.org/abs/2103.03230)
- InfoNCE/SimCLR (https://arxiv.org/abs/2002.05709)
- Temperature as Uncertainty (TaU) (https://arxiv.org/abs/2110.04403)

# Augmentation Policies

- Default (from VICReg paper: https://arxiv.org/abs/2105.04906)

# Datasets

- ImageNet (https://ieeexplore.ieee.org/document/5206848)
- CIFAR10 (https://www.cs.toronto.edu/~kriz/cifar.html)

# Model Zoo Support

- timm (https://github.com/huggingface/pytorch-image-models)

# Under Active Development

- Probabilistic Contrastive Learning (https://arxiv.org/abs/2302.02865)


