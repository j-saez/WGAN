import os
if os.getcwd()[-4:] != 'wgan':
    raise Exception('Run the file from the root directory.')

###########
# imports #
###########

import torchvision
from torch.utils.data import Dataset

AVAILABLE_DATASETS = ['mnist']
DATASETS_PATH = os.getcwd() + '/datasets/'

"""
    Name: get_dataset_transforms
    Description: Returns the transformations to be applied to the dataset.
    Inputs:
        >> img_size: (tuple of ints) Height and width of the images in the dataset
        >> img_ch: (int) Number of channels in the images of the dataset.
    Outputs:
        >> transforms: (torchvision.transforms) Tranformations to be applied to the images.
"""
def get_dataset_transforms(img_size, img_ch):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size), 
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize([0.5 for _ in range(img_ch)],[0.5 for _ in range(img_ch)]), 
    ])
    return transforms

"""
    Name: load_dataset
    Description: Return the train and test datasets of the selected dataset.
    Inputs:
        >> dataset_name: (str) Name of the dataset to be loaded.
        >> transforms: (torchvision.transforms) transformations to apply to the dataset when loading.
    Outputs:
        >> train_dataset: (torch.utils.data.Dataset) Train data.
        >> test_dataset: (torch.utils.data.Dataset) Test data.
"""
def load_dataset(dataset_name, transforms):
    train_dataset = Dataset()
    if not dataset_name in AVAILABLE_DATASETS:
        raise Exception(f'{dataset_name} not available. The avalable ones are: {AVAILABLE_DATASETS}')
    if dataset_name == 'mnist':
        train_dataset = torchvision.datasets.MNIST(root=DATASETS_PATH,train=True,transform=transforms,download=True)
    return train_dataset
