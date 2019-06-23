"""
Created on 22/06/2019
@author: nidragedd
"""
import argparse
from pathlib import Path

from src.utils import constants


def define_train_input_parameters():
    """
    Build all arguments needed for the "training" script
    :return: parsed arguments
    """
    # Mandatory (or not) arguments to run the program
    ap = argparse.ArgumentParser()
    ap.add_argument('data', help='path to training dataset directory')
    ap.add_argument("-a", "--arch", required=True, help="Name of the model, must be a valid choice")
    ap.add_argument("-o", "--save_dir", required=True,
                    help="Output directory where the model will be saved once trained")
    ap.add_argument("-lr", "--learning_rate", required=False, help="Set the learning rate, default is 0.001")
    ap.add_argument("-hu", "--hidden_units", required=False, help="Set the number of hidden units for the classifier")
    ap.add_argument("-e", "--epochs", required=False, help="Set the TOTAL number of epochs to train on, default is 2")
    ap.add_argument("-gpu", "--gpu", required=False, action='store_true',
                    help="If specified, training will be done on GPU")

    return vars(ap.parse_args())


def define_predict_input_parameters():
    """
    Build all arguments needed for the "predict" script
    :return: parsed arguments
    """
    # Mandatory (or not) arguments to run the program
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Full path to an image to classify")
    ap.add_argument("checkpoint", help="Checkpoint file containing all data saved after training")
    ap.add_argument("-t", "--top_k", required=False, help="How many most likely classes to display")
    ap.add_argument("-c", "--category_names", required=False, help="Path to JSON file containing")
    ap.add_argument("-gpu", "--gpu", required=False, action='store_true',
                    help="If specified, training will be done on GPU")

    return vars(ap.parse_args())

def get_training_dirs(args):
    """
    Given all script arguments, perform some sanity checks then build train, test and validation directories
    :param args: the given script arguments (already parsed)
    :return: (tuple) paths to train, validation and test directories, in this order
    """
    if args["arch"] not in constants.ALLOWED_MODELS_NAMES:
        raise Exception("Model must be a choice between those values: {}".format(constants.ALLOWED_MODELS_NAMES))

    data_dir = Path(args["data"])
    train = data_dir.joinpath('train')
    valid = data_dir.joinpath('valid')
    test = data_dir.joinpath('test')
    if not data_dir.exists():
        raise FileExistsError('The specified data directory "{}" does not exist'.format(data_dir))
    if not train.exists():
        raise FileExistsError('The mandatory "train" directory does not exist in the given dataset directory !')
    if not valid.exists():
        raise FileExistsError('The mandatory "valid" directory does not exist in the given dataset directory !')
    if not test.exists():
        raise FileExistsError('The mandatory "test" directory does not exist in the given dataset directory !')

    return train, valid, test
