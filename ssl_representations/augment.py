from PIL import ImageOps, ImageFilter
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from typing import Tuple


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class TrainTransform(object):
    r"""Performs two augmentations on the same batch of samples.
    The augmentation policy is critical in the performance of SSL methods.
    The role of augmentations has been investigated in https://arxiv.org/abs/2106.04619
    and https://arxiv.org/abs/2302.02774.
    
    Note that the prime transform is slightly different from the normal
    transform. The normal transform is guaranteed to use Gaussian Blur,
    while the primed transform has only a 10% chance. The normal transform
    also does not solarize, while the primed transform has a 0.2 chance to solarize.

    Parameters
    ----------
    sample: torch.Tensor

        A batch of samples from the dataset.

    Returns
    -------
    x1: torch.Tensor

        A randomly augmented batch of samples.
    
    x2: torch.Tensor
        
        A second randomly augmented batch of samples. x1[i] and x2[i] correspond to the same sample.
    """
    def __init__(self, args):
        r"""Initializes the augmentation policy as given by args, which is passed from the parser in the train file.

        Currently, only the augmentation policy from VICReg is implemented, which consists of:

        RandomResizeCrop
        RandomHorizontalFlip
        ColorJitter
        RandomGrayscale
        GaussianBlur
        Solarization
        Normalize
        """
        self.args = args
        self.resolution = None

        if self.args.dataset == 'imagenet':
            
            self.resolution = 224
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        
        elif self.args.dataset == 'cifar10':
            
            self.resolution = 32
            self.mean = [0.49139968, 0.48215827 ,0.44653124]
            self.std = [0.24703233, 0.24348505, 0.26158768]

        else:
            raise NotImplementedError("Dataset has not yet been implemented.")

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.resolution, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean, std=self.std
                ),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.resolution, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean, std=self.std
                ),
            ]
        )

    def __call__(self, sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2