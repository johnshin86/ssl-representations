import torchvision.transforms as T
import torch

class VOC_preprocess:
    r"""The preprocess class for the PACAL VOC 2012 dataset. This is designed to be used in the dataset class at instantiation,
    such that all samples drawn from the dataset are preprocessed in this manner, whether or not they're further
    augmented. 
    """
    def __init__(self, data_preprocess: str):

        self.data_preprocess = data_preprocess
        #Mean and std of the dataset computed with (224,224) center cropping.
        self.mean = torch.tensor([0.4570, 0.4382, 0.4062])
        self.std = torch.tensor([0.2345, 0.2305, 0.2353])

        if data_preprocess == "normalize":
            self.transforms = T.Compose([
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std),
                    T.Resize((224, 224))
            ])
        else:
            self.transforms = T.Compose([
                    T.ToTensor(),
                    T.Resize((224, 224))
            ])

    def __call__(self, img):
        return self.transforms(img)

class VOC_augment:
    r"""The augment class for the PASCAL VOC 2012 dataset. This is designed to be used to augment the dataset during
    training for non-contrastive self-supervised learning methods. 
    """
    def __init__(self, aug_policy: str, aug_strength: str):

        self.aug_policy = aug_policy
        self.aug_strength = aug_strength

        self.RandomJitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p = 0.8)
        self.RandomGrayscale = T.RandomApply([T.Grayscale(num_output_channels = 3)], p = 0.2)
        self.RandomGaussian = T.RandomApply([T.GaussianBlur(kernel_size = 3)], p = 0.5)

        if self.aug_policy == 'standard':


            if self.aug_strength == 'weak':

                self.transforms = T.Compose([
                    self.RandomGaussian,
                    ])

            elif self.aug_strength == 'medium':

                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(p = 0.5),
                    self.RandomJitter,
                    self.RandomGrayscale,
                    self.RandomGaussian,
                    ])

            elif self.aug_strength == 'strong':
                self.transforms = T.Compose([
                    T.RandomResizedCrop(size = (224,224), scale = (0.05, 1.0))
                    T.RandomHorizontalFlip(p = 0.5),
                    self.RandomJitter,
                    self.RandomGrayscale,
                    self.RandomGaussian,
                    T.RandomSolarize(p = 0.1),
                    T.Normalize(mean, std),
                    ])

            else:
                raise ValueError(f'Unknown augmentation strength "{aug_strength}"')

                

        else:
            raise ValueError(f'Unknown data augmentation policy "{aug_policy}"')

    def __call__(self, img):
        return self.transforms(img)