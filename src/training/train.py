"""
Package used to hold all methods related to training steps

Created on 22/06/2019
@author: nidragedd
"""
import time

from src.performance import score
from src.utils import constants


def train_model(device, model, start_epoch, total_epochs, trainloader, validationloader, criterion, optimizer):
    """
    Train the given model for a given number of epochs
    :param device: (string) the device where to put model, images, labels (might be GPU for example)
    :param model: (object) the torch model object that will be evaluated
    :param start_epoch: (int) to use only if we want to resume training from a previous run
    :param total_epochs: (int) the total number of epochs to train on
    :param trainloader: (object) the dataloader containing training images
    :param validationloader: (object) the dataloader containing validation images
    :param criterion: (object) torch criterion object to use to compute a loss
    :param optimizer: (object) torch optimizer object to use to optimize the loss
    :return: (int) the number of epochs this model has been trained on (overall runs)
    """
    model.to(device)

    steps = 0
    running_loss = 0
    start = time.time()

    print('Training is starting...')

    for epoch in range(start_epoch, total_epochs):
        model.train()
        for images, labels in iter(trainloader):

            steps += 1
            # Move input and label tensors to the right device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % constants.train_print_every == 0:
                valid_loss, valid_accuracy = score.get_metrics(device, model, validationloader, criterion)

                print("Epoch: {}/{}.. ".format(epoch + 1, total_epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / constants.train_print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss / len(validationloader)),
                      "Validation Accuracy: {:.2f} %".format(valid_accuracy))

                running_loss = 0

    end = time.time()
    print('Training finished ! Time taken for whole training is {:.2f} seconds'.format(end - start))
    return epoch
