import os
import utils as myutils
import sys
PATH_TO_CIFAR = "./cifar/"
sys.path.append(PATH_TO_CIFAR)
import train as cifar_train
import hyperparameters.vgg11_cifar10_baseline as vgg_hyperparams
import wasserstein_ensemble
import baseline
import parameters
import torch
ensemble_root_dir = "./cifar_models/"
# ensemble_experiment = "exp_2019-04-23_18-08-48/"
ensemble_experiment = "exp_2019-04-24_02-20-26"
ensemble_dir = ensemble_root_dir + ensemble_experiment

output_root_dir = "./cifar_models_ensembled/"
checkpoint_type = 'final' # which checkpoint to use for ensembling (either of 'best' or 'final)

def main():
    # torch.cuda.empty_cache()
    config = vgg_hyperparams.config
    timestamp = myutils.get_timestamp_other()

    model_list = os.listdir(ensemble_dir)
    num_models = len(model_list)

    train_loader, test_loader = cifar_train.get_dataset(config)

    models = []

    for idx in range(num_models):
        print("Path is ", ensemble_dir)
        print("loading model with idx {} and checkpoint_type is {}".format(idx, checkpoint_type))
        models.append(
            cifar_train.get_pretrained_model(
                config, os.path.join(ensemble_dir, 'model_{}/{}.checkpoint'.format(idx, checkpoint_type)), parameters.gpu_id
            )
        )

    print("Done loading all the models")

    # run geometric aka wasserstein ensembling
    print("------- Geometric Ensembling -------")
    wasserstein_ensemble.geometric_ensembling_modularized(models, train_loader, test_loader)

    # run baseline
    print("------- Prediction based ensembling -------")
    baseline.prediction_ensembling(models, train_loader, test_loader)
    print("------- Naive ensembling of weights -------")

    baseline.naive_ensembling(models, train_loader, test_loader)


if __name__ == '__main__':
    main()