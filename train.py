"""
Created on 22/06/2019
@author: nidragedd
"""
import torch
from torch import nn
from torch import optim

from pathlib import Path

from src.performance import score
from src.training import build, train
from src.utils import utils, constants, load_restore
from src.preprocess import preprocess

if __name__ == "__main__":
    args = utils.define_train_input_parameters()

    output_dir = Path(args["save_dir"])
    output_dir.mkdir(exist_ok=True)
    if not output_dir.exists():
        raise FileExistsError('The output save directory does not exist nor could be created')

    train_dir, valid_dir, test_dir = utils.get_training_dirs(args)
    train_dataloader, valid_dataloader, test_dataloader = preprocess.transform_and_build_dataloader(train_dir,
                                                                                                    valid_dir, test_dir)

    # Other parameters management (either specified or default values
    # Enabling GPU if specified AND possible
    device = 'cpu'
    if args['gpu']:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args['arch']
    learning_rate = args['learning_rate'] if args['learning_rate'] else constants.default_lr
    hidden_units = args['hidden_units'] if args['hidden_units'] else constants.default_hidden_units
    total_epochs = args['epochs'] if args['epochs'] else constants.default_total_epochs
    start_epoch = 0  # No resume handled so far but maybe later

    # BUILD MODEL, LOSS & OPTIMIZER
    model = build.build_model(model_name, hidden_units)
    # Since the model's forward method returns the log-softmax, use the negative log loss as criterion
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    epochs_done = train.train_model(device, model, start_epoch, total_epochs, train_dataloader, valid_dataloader,
                                    criterion, optimizer)

    # VALIDATION PHASE on the test set
    _, accuracy = score.get_metrics(device, model, test_dataloader)
    print("Test dataset accuracy: {:.2f} %".format(accuracy))

    # SAVE TRAINED MODEL
    savefile = output_dir.joinpath('checkpoint-script.pth')
    class_to_idx = train_dataloader.dataset.class_to_idx
    load_restore.save_checkpoint(epochs_done, hidden_units, model_name, model, class_to_idx, savefile)
    print("Model successfully saved on disk at this place {}".format(savefile))
