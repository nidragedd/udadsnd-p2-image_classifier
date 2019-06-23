"""
Created on 22/06/2019
@author: nidragedd
"""
import torch
import numpy as np
import json

from pathlib import Path

from src.preprocess import preprocess
from src.utils import utils, constants, load_restore, display


def predict(image_path, model, class2idx, topk=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    :param image_path: (string) full path to the image we have to classify
    :param model: (torch model) the loaded and previously trained pytorch model
    :param topk: (int) how many 'most likely' classes to get
    :return: (tuple) first element is an array of probabilities, second one is and array of classes number corresponding
    to the most likely classes the model has predicted
    """
    input_img = preprocess.process_image(image_path)
    torch_input_img = torch.from_numpy(input_img).float()
    torch_input_img.unsqueeze_(0)  # Add a dimension at first position (will represent the batch size, here 1)

    model.eval()
    # Turn off gradients for validation --> saves memory and computations
    with torch.no_grad():
        output = model.forward(torch_input_img)
        probs, indices = output.topk(topk)
        # Transform tensors to numpy arrays and take the first (and single) element
        probs = np.exp(probs.numpy()[0])  # Do not forget to get back the exponential value as output is log-softmax !
        indices = indices.numpy()[0]
        # Revert the dict 'class to indice' to get 'indice to class'
        idx_classes = {v: k for k, v in class2idx.items()}
        classes = [v for k, v in idx_classes.items() if k in indices]

    return probs, classes


if __name__ == "__main__":
    args = utils.define_predict_input_parameters()

    input_img = Path(args["input"])
    checkpoint = Path(args["checkpoint"])
    if not input_img.exists():
        raise FileExistsError('The input image does not exist !')
    if not checkpoint.exists():
        raise FileExistsError('The checkpoint file does not exist !')

    # Other parameters management (either specified or default values
    # Enabling GPU if specified AND possible
    device = 'cpu'
    if args['gpu']:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cat_to_name = None
    if args['category_names']:
        json_file = Path(args['category_names'])
        if not json_file.exists():
            raise FileExistsError('The given categories to name JSON file does not exist !')
        else:
            with open(json_file, 'r') as f:
                cat_to_name = json.load(f)
    top_k = args['top_k'] if args['top_k'] else constants.top_k

    # LOADING MODEL FROM CHECKPOINT + prediction
    model, nb_epochs_done, class2idx = load_restore.load_checkpoint(checkpoint)
    probs, classes = predict(input_img, model, class2idx, int(top_k))

    # Display the most K likely classes and their probabilities
    classnames = classes
    if cat_to_name:
        classnames = [cat_to_name[i] for i in classes]
    display.display_most_likely_classes(input_img, classnames, probs)
