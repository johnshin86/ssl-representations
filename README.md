# SSL Representations

This library implements many recent vision SSL methods, as well as uncertainty estimation methods built on top of these methods. 

Self-supervised learning in vision is an interplay of the augmentation policy, the inductive bias of the architecture,
and the interactions across samples induced by the loss function (https://arxiv.org/abs/2302.02774). The library was initially built from the facebook VICReg repo,
but has been heavily modified.

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


