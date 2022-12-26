import torchvision.transforms as T
import torch
from sen12ms import S1Bands, S2Bands, LCBands, Seasons, Sensor

class sen12ms_preprocess:
    r"""The preprocess class for the sen12ms dataset. This is designed to be used as the dataset class is instantiation,
    such that all samples drawn from the dataset are preprocessed in this manner, whether or not they're further
    augmented. 
    """
    def __init__(self, data_preprocess: str, bands = None):

        if bands == S2Bands.RGB:
            self.mean = torch.tensor([0.1126, 0.1128, 0.1218], dtype=torch.float64)
            self.std = torch.tensor([0.0264, 0.0174, 0.0137], dtype=torch.float64)
        elif bands == S1Bands.ALL:
            # Note: the statistics for SAR were calculated by zero'ing Nans.
            self.mean = torch.tensor([-1500.8836, -2383.4999], dtype=torch.float64)
            self.std = torch.tensor([382.4406, 395.6070], dtype=torch.float64)
        elif bands == S1Bands.VV:
            self.mean = torch.tensor([-1500.8836], dtype=torch.float64)
            self.std = torch.tensor([382.4406], dtype=torch.float64)
        elif bands == S1Bands.VH:
            self.mean = torch.tensor([-2383.4999], dtype=torch.float64)
            self.std = torch.tensor([395.6070], dtype=torch.float64)

        else:
            raise ValueError(f'Unimplemented band type "{bands}" for  policy"{data_preprocess}"')


        if data_preprocess == 'normalize':
            self.transforms = T.Compose([
                #T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ])
        else:
            raise ValueError(f'Unknown data preprocessing policy "{data_preprocess}"')

    def __call__(self, img):
        return self.transforms(img)

class sen12flood_preprocess:
    r"""The preprocess class for the sen12flood dataset. This is designed to be used as the dataset class is instantiation,
    such that all samples drawn from the dataset are preprocessed in this manner, whether or not they're further
    augmented. 
    """
    def __init__(self, data_preprocess: str, bands = None):

        if bands == S2Bands.ALL:
            self.mean = torch.tensor([0.1265, 0.1162, 0.0873], dtype=torch.float64)
            self.std = torch.tensor([0.0716, 0.0598, 0.0564], dtype=torch.float64)
        #TODO: SAR data is not the same dimension.
        elif bands == S1Bands.ALL:
            raise ValueError(f'Unimplemented band type "{bands}" for  policy"{data_preprocess}"')
        else:
            raise ValueError(f'Unimplemented band type "{bands}" for  policy"{data_preprocess}"')


        if data_preprocess == 'normalize':
            self.transforms = T.Compose([
                #T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ])
        else:
            raise ValueError(f'Unknown data preprocessing policy "{data_preprocess}"')

    def __call__(self, img):
        return self.transforms(img)

class sen12ms_augment:
    r"""The augment class for the sen12ms dataset. This is designed to be used to augment the dataset during
    training for non-contrastive self-supervised learning methods. 
    """
    def __init__(self, data_augmentation: str, bands = None):
        self.RandomJitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p = 0.8)
        self.RandomGrayscale = T.RandomApply([T.Grayscale(num_output_channels = 3)], p = 0.2)
        self.RandomGaussian = T.RandomApply([T.GaussianBlur(kernel_size = 3)], p = 0.5)

        if data_augmentation == 'vicreg':

            if bands == S2Bands.RGB:
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(p = 0.5),
                    self.RandomJitter,
                    self.RandomGrayscale,
                    self.RandomGaussian,
                    #T.RandomSolarize(p = 0.1),
                    ])
            elif bands == S1Bands.ALL:
                # TODO: add more augments for SAR (2 channel).
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(p = 0.5),
                    #self.RandomJitter,
                    #self.RandomGrayscale,
                    self.RandomGaussian,
                    #T.RandomSolarize(p = 0.1),
                    ])
            elif bands == S1Bands.VV:
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(p = 0.5),
                    self.RandomJitter,
                    #self.RandomGrayscale,
                    self.RandomGaussian,
                    #T.RandomSolarize(p = 0.1),
                    ])
            elif bands == S1Bands.VH:
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(p = 0.5),
                    self.RandomJitter,
                    #self.RandomGrayscale,
                    self.RandomGaussian,
                    #T.RandomSolarize(p = 0.1),
                    ])
            else:
                raise ValueError(f'Unimplemented band type "{bands}" for  policy"{data_preprocess}"')

        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

    def __call__(self, img):
        return self.transforms(img)