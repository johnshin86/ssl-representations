import torchvision.transforms as T
import torch

class VOC_preprocess:
    r"""The preprocess class for the PACAL VOC 2012 dataset. This is designed to be used in the dataset class at instantiation,
    such that all samples drawn from the dataset are preprocessed in this manner, whether or not they're further
    augmented. 
    """
    def __init__(self, data_preprocess: str):

        self.transforms = T.Compose([
                T.ToTensor(),
        ])

    def __call__(self, img):
        return self.transforms(img)

class VOC_augment:
    r"""The augment class for the PASCAL VOC 2012 dataset. This is designed to be used to augment the dataset during
    training for non-contrastive self-supervised learning methods. 
    """
    def __init__(self, data_augmentation: str):
        self.RandomJitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p = 0.8)
        self.RandomGrayscale = T.RandomApply([T.Grayscale(num_output_channels = 3)], p = 0.2)
        self.RandomGaussian = T.RandomApply([T.GaussianBlur(kernel_size = 3)], p = 0.5)

        if data_augmentation == 'vicreg':

            self.transforms = T.Compose([
                    T.RandomHorizontalFlip(p = 0.5),
                    self.RandomJitter,
                    self.RandomGrayscale,
                    self.RandomGaussian,
                    #T.RandomSolarize(p = 0.1),
                    ])
                

        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

    def __call__(self, img):
        return self.transforms(img)