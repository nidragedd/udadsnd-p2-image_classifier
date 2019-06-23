"""
Package used to hold all methods related to building models

Created on 22/06/2019
@author: nidragedd
"""
from torch import nn
from torchvision import models

from collections import OrderedDict
from src.utils import constants


def _extend_pretrained_network(model, in_features, hidden_layers):
    """
    Extend an existing and pretrained network by replacing its classifier only (==> transfer learning)
    Note that model parameters are frozen so there will no backpropagation through them
    :param model: (object) the torch model object to extend
    :param in_features: (int) the number of features the classifier takes as input (depends on the chosen model)
    :param hidden_layers: (int) the number of hidden layers to put in our specific classifier
    :return: (object) the updated (extended) torch model
    """
    for param in model.parameters():
        param.requires_grad = False

    # Build a new classifier that will replace the existing one in the pretrained model
    # Take care of the size for input on this classifier, it might change depending on the chosen model
    # Note: as seen in the course, we are going to use the log-softmax as output which is a log probability
    #       Using the log probability, computations are often faster and more accurate. To get the class probabilities
    #       later, we will need to take the exponential (torch.exp) of the output
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features, hidden_layers)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_layers, constants.nb_classes_output)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    # Change the model's classifier: careful ! depending on the network, the name of the classifier might change...
    model.classifier = classifier

    return model


def build_model(model_name, hidden_layers):
    """
    Build a pretrained torch model object and extend it
    :param model_name: (string) a model name
    :param hidden_layers: (int) the number of hidden layers to put in our specific classifier
    :return: (object) the updated (extended) torch model
    """
    # Hard-coded cases, goal is not to have something more beautiful
    model = None
    if model_name == constants.PRETRAINED_RESNET:
        model = models.resnet50(pretrained=True)
    elif model_name == constants.PRETRAINED_VGG16:
        model = models.vgg16(pretrained=True)
    elif model_name == constants.PRETRAINED_DENSENET:
        model = models.densenet121(pretrained=True)

    in_features = constants.model_clf_inputs[model_name]
    model = _extend_pretrained_network(model, in_features, hidden_layers)

    return model
