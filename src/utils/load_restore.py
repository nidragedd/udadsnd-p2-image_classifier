"""
Package used to hold all methods related to model serialization or loading from disk

Created on 22/06/2019
@author: nidragedd
"""
import torch

from src.training import build
from src.utils import constants


def save_checkpoint(epochs_done, hidden_units, model_name, model, class2idx, filename='checkpoint.pth'):
    """
    Basically save the model checkpoints
    :param epochs_done: (int) number of epochs already done for training
    :param hidden_units: (int) number of hidden units
    :param model_name: (string) name of the chosen model
    :param model: (object) torch model object to save
    :param class2idx: (dict) class to indices dict based on training dataloaders
    :param filename: (string) the checkpoint filename if specified
    """
    checkpoint = {
        constants.ls_nb_epochs_done: epochs_done + 1,
        constants.ls_hidden_units: hidden_units,
        constants.ls_model_name: model_name,
        constants.ls_state_dict: model.state_dict(),
        constants.ls_class2idx: class2idx
    }
    torch.save(checkpoint, filename)


def load_checkpoint(filename):
    """
    Load a checkpoint file and then rebuilds the model
    :param filename: (string) the checkpoint filename
    :return: (tuple) first element is the torch model rebuilt from checkpoint file, second is an int representing the
    number of epochs already done (can be used to resume training), third is the class to indice element that will be
    used for classification
    """
    # Based on this discussion (https://discuss.pytorch.org/t/problem-loading-model-trained-on-gpu/17745/3) I had to add
    # map_location option to be able to load a model trained on GPU to then use it on CPU
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    nb_epochs_done = checkpoint[constants.ls_nb_epochs_done]
    hidden_units = checkpoint[constants.ls_hidden_units]
    class2idx = checkpoint[constants.ls_class2idx]

    model = build.build_model(checkpoint[constants.ls_model_name], hidden_units)
    model.load_state_dict(checkpoint[constants.ls_state_dict])

    return model, nb_epochs_done, class2idx
