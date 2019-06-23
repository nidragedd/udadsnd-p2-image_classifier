"""
Package used to hold all methods related to preprocessing steps

Created on 22/06/2019
@author: nidragedd
"""
import matplotlib
# Set the matplotlib backend to a non-interactive one so figures can be saved in the background
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.preprocess import preprocess
from src.utils import constants


def imshow(image, ax=None):
    """
    Image show given a tensor as input
    :param image: (torch tensor) the image data as a Torch tensor
    :param ax: (matplotlib Axis) not required, will be used if given or created otherwise
    :return: (matplotlib Axis)
    """
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array(constants.norm_means)
    std = np.array(constants.norm_std)
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def display_most_likely_classes(image_path, classnames, probs):
    """
    Display an image along with the top K classes
    :param image_path: (string) full path to the image to display
    :param classnames: (array) human readable names for most likely predicted classes
    :param probs: (array) values for classes probabilities
    """
    plt.style.use("ggplot")
    figure, axis = plt.subplots(2, 1, figsize=(15, 10))
    axis[0].set_title(classnames[0])
    axis[0].set_axis_off()
    axis[1].barh(np.arange(len(probs)), probs, tick_label=classnames)
    axis[1].set_aspect(0.1)
    axis[1].invert_yaxis()

    imshow(torch.from_numpy(preprocess.process_image(image_path)), axis[0])
    plt.savefig("prediction.png")
