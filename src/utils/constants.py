"""
All constants (or default values are put here).
Thus, later and if needed, all this configuration can be externalized in a config.json file for example

Created on 22/06/2019
@author: nidragedd
"""

# The number of classes that exists in this classification problem
nb_classes_output = 102

# ########################################
# #############    MODELS     ############
# ########################################
PRETRAINED_VGG16 = 'vgg16'
PRETRAINED_DENSENET = 'densenet121'
PRETRAINED_RESNET = 'resnet50'
ALLOWED_MODELS_NAMES = [PRETRAINED_VGG16, PRETRAINED_DENSENET, PRETRAINED_RESNET]
# Define a dict where key is a model name and value is the number of features the classifier takes as input
model_clf_inputs = {
    'resnet50': 2048,
    'densenet121': 1024,
    'vgg16': 25088
}

# ########################################
# ##########  PREPROCESSING     ##########
# ########################################
norm_means = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
resizing_size = 256
cropping_size = 224
batch_size = 64

# ########################################
# ##########      TRAINING     ###########
# ########################################
train_print_every = 10
default_lr = 0.001
default_hidden_units = 500
default_total_epochs = 2

# ########################################
# ##########     SAVE/LOAD     ###########
# ########################################
ls_nb_epochs_done = 'nb_epochs_done'
ls_model_name = 'model_name'
ls_hidden_units = 'hidden_units'
ls_state_dict = 'state_dict'
ls_class2idx = 'class2idx'

# ########################################
# #########     PREDICTION     ###########
# ########################################
top_k = 5
