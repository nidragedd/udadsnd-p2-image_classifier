"""
Package used to hold all methods related to performance measures (accuracy, whatever)

Created on 22/06/2019
@author: nidragedd
"""
import torch


def get_metrics(device, model, dataloader, criterion=None):
    """
    Method that provides accuracy metric + a loss if criterion is provided in input argument on the given dataloader
    :param device: (string) the device where to put model, images, labels (might be GPU for example)
    :param model: (object) the torch model object that will be evaluated
    :param dataloader: (object) the dataloader containing images (might be the test or the validation one depending on
    the usage)
    :param criterion: (object) not required but if given then this method will compute and return a loss value
    :return: (tuple) loss (default is 0 is no criterion argument given), accuracy metric (how many well classified
    images on the given dataloader)
    """
    loss = 0
    correct = 0
    total = 0
    model.eval()  # Make sure network is in eval mode for inference

    # Turn off gradients for validation --> saves memory and computations
    with torch.no_grad():
        for images, labels in dataloader:
            # Move input and label tensors to the right device
            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
            if criterion:
                loss += criterion(output, labels).item()

            total += labels.size(0)  # = batch size

            ps = torch.exp(output)  # Remember, the output is a log softmax ! So take the exp to get back the original

            # ps has shape(batch_size, nb_classes) so we take the index for which the computed value is the max among
            # all classes probabilities and we compare this to the ground truth which is labels.data
            # it then gives us a tensor of boolean that we can sum over to get the number of correctly classified images
            equality = (labels.data == ps.max(dim=1)[1])

            # Sum the number of correctly classified images among the given dataset
            correct += equality.type(torch.FloatTensor).sum().item()

    accuracy = 100 * correct / total

    model.train()  # Make sure training is back on

    return loss, accuracy
