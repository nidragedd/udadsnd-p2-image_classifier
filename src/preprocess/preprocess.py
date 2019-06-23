"""
Package used to hold all methods related to preprocessing steps

Created on 22/06/2019
@author: nidragedd
"""
import numpy as np
from torch.utils.data import  DataLoader
from torchvision import datasets, transforms
from PIL import Image

from src.utils import constants


def transform_and_build_dataloader(train_dir, valid_dir, test_dir):
    """
    Define transforms for the training, validation, and testing sets.
    :param train_dir: (path) path to the training directory
    :param valid_dir: (path) path to the validation directory
    :param test_dir: (path) path to the testing directory
    :return: (tuple) a tuple of the 3 dataloaders built for each of the given directories
    """
    # Data augmentation for the training set
    train_trans = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(constants.cropping_size),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(constants.norm_means, constants.norm_std)])

    # No data augmentation for validation or test datasets, just resizing and center+crop (+ same normalization)
    test_valid_trans = transforms.Compose([transforms.Resize(constants.resizing_size),
                                           transforms.CenterCrop(constants.cropping_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(constants.norm_means, constants.norm_std)])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_trans)
    test_data = datasets.ImageFolder(test_dir, transform=test_valid_trans)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_trans)

    # Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = DataLoader(train_data, batch_size=constants.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=constants.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=constants.batch_size, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader


def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model
    :param image: (string) full path to the image to process
    :return: (ndarray)
    """
    new_size = constants.cropping_size

    im = Image.open(image)
    # Resize but keeping aspect ratio
    im.thumbnail((constants.resizing_size, constants.resizing_size))

    # Center crop to 224
    width, height = im.size  # Get new dimensions
    left = int(np.ceil((width - new_size) / 2))
    right = width - int(np.floor((width - new_size) / 2))
    top = int(np.ceil((height - new_size) / 2))
    bottom = height - int(np.floor((height - new_size) / 2))
    im = im.crop((left, top, right, bottom))

    mean = np.array(constants.norm_means)
    std = np.array(constants.norm_std)

    # Scale the raw pixel intensities to the range [0, 1]
    np_image = np.array(im, dtype="float") / 255.0
    # Colors normalization
    np_image = (np_image - mean) / std

    # pytorch expects color channel to be in first position ! --> use transpose
    return np_image.transpose((2, 0, 1))
