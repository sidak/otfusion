import parameters
from data import get_dataloader
import routines
import baseline
import wasserstein_ensemble
import os
import utils
import numpy as np
import sys
import hyperparameters.vgg11_cifar10_baseline as vgg_hyperparams
PATH_TO_CIFAR = "./cifar/"
sys.path.append(PATH_TO_CIFAR)
import train as cifar_train


exp_path = sys.argv[1]
gpu_id = int(sys.argv[2])
print("gpu_id is ", gpu_id)
print("exp_path is ", exp_path)

config = vgg_hyperparams.config

model_types = ['model_0', 'model_1', 'geometric', 'naive_averaging']
for model in model_types:
    for ckpt in ['best', 'final']:
        if os.path.exists(os.path.join(exp_path, model)):
            cifar_train.get_pretrained_model(config,
                os.path.join(exp_path, model, ckpt + '.checkpoint'), device_id=gpu_id)
