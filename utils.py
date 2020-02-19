import os
import pickle
from itertools import chain
import csv
import collections
import numpy as np
import ot
import sys
PATH_TO_CIFAR = "./cifar/"
sys.path.append(PATH_TO_CIFAR)
import train as cifar_train
PATH_TO_VGG = "./cifar/models/"
sys.path.append(PATH_TO_VGG)
import vgg
import partition

def get_timestamp_other():
    import time
    import datetime
    ts = time.time()
    # %f allows granularity at the micro second level!
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S_%f')
    return timestamp

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def mkdir(path):
    os.makedirs(path, exist_ok=True)
    # if not os.path.exists(path):
    #     os.makedirs(path)

def pickle_obj(obj, path, mode = "wb", protocol=pickle.HIGHEST_PROTOCOL):
    '''
    Pickle object 'obj' and dump at 'path' using specified
    'mode' and 'protocol'
    Returns time taken to pickle
    '''

    import time
    st_time = time.perf_counter()
    pkl_file = open(path, mode)
    pickle.dump(obj, pkl_file, protocol=protocol)
    end_time = time.perf_counter()
    return (end_time - st_time)

def dict_union(*args):
    return dict(chain.from_iterable(d.items() for d in args))

def save_results_params_csv(path, results_dic, args, ordered=True):
    if os.path.exists(path):
        add_header = False
    else:
        add_header = True

    with open(path, mode='a') as csv_file:

        if args.deprecated is not None:
            params = args
        else:
            params = vars(args)

        # Merge with params dic
        if ordered:
            # sort the parameters by name before saving
            params = collections.OrderedDict(sorted(params.items()))

        results_and_params_dic = dict_union(results_dic, params)

        writer = csv.DictWriter(csv_file, fieldnames=results_and_params_dic.keys())

        # Add key header if file doesn't exist
        if add_header:
            writer.writeheader()

        # Add results and params record
        writer.writerow(results_and_params_dic)

def isnan(x):
    return x != x


def get_model_activations(args, models, config=None, layer_name=None, selective=False, personal_dataset = None):
    import compute_activations
    from data import get_dataloader

    if args.activation_histograms and args.act_num_samples > 0:
        if args.dataset == 'mnist':
            unit_batch_train_loader, _ = get_dataloader(args, unit_batch=True)

        elif args.dataset.lower()[0:7] == 'cifar10':
            if config is None:
                config = args.config # just use the config in arg
            unit_batch_train_loader, _ = cifar_train.get_dataset(config, unit_batch_train=True)

        if args.activation_mode is None:
            activations = compute_activations.compute_activations_across_models(args, models, unit_batch_train_loader,
                                                                        args.act_num_samples)
        else:
            if selective and args.update_acts:
                activations = compute_activations.compute_selective_activation(args, models,
                                                                               layer_name, unit_batch_train_loader,
                                                                               args.act_num_samples)
            else:
                if personal_dataset is not None:
                    # personal training set is passed which consists of (inp, tgts)
                    print('using the one from partition')
                    loader = partition.to_dataloader_from_tens(personal_dataset[0], personal_dataset[1], 1)
                else:
                    loader = unit_batch_train_loader

                activations = compute_activations.compute_activations_across_models_v1(args, models,
                                                                                       loader,
                                                                                       args.act_num_samples,
                                                                                       mode=args.activation_mode)

    else:
        activations = None

    return activations

def get_model_layers_cfg(model_name):
    print('model_name is ', model_name)
    if model_name == 'mlpnet' or model_name[-7:] =='encoder':
        return None
    elif model_name[0:3].lower()=='vgg':
        cfg_key = model_name[0:5].upper()
    elif model_name[0:6].lower() == 'resnet':
        return None
    return vgg.cfg[cfg_key]


def _get_config(args):
    print('refactored get_config')
    import hyperparameters.vgg11_cifar10_baseline as cifar10_vgg_hyperparams  # previously vgg_hyperparams
    import hyperparameters.vgg11_half_cifar10_baseline as cifar10_vgg_hyperparams_half
    import hyperparameters.vgg11_doub_cifar10_baseline as cifar10_vgg_hyperparams_doub
    import hyperparameters.vgg11_quad_cifar10_baseline as cifar10_vgg_hyperparams_quad
    import hyperparameters.resnet18_nobias_cifar10_baseline as cifar10_resnet18_nobias_hyperparams
    import hyperparameters.resnet18_nobias_nobn_cifar10_baseline as cifar10_resnet18_nobias_nobn_hyperparams
    import hyperparameters.mlpnet_cifar10_baseline as mlpnet_hyperparams

    config = None
    second_config = None

    if args.dataset.lower() == 'cifar10':
        if args.model_name == 'mlpnet':
            config = mlpnet_hyperparams.config
        elif args.model_name == 'vgg11_nobias':
            config = cifar10_vgg_hyperparams.config
        elif args.model_name == 'vgg11_half_nobias':
            config = cifar10_vgg_hyperparams_half.config
        elif args.model_name == 'vgg11_doub_nobias':
            config = cifar10_vgg_hyperparams_doub.config
        elif args.model_name == 'vgg11_quad_nobias':
            config = cifar10_vgg_hyperparams_quad.config
        elif args.model_name == 'resnet18_nobias':
            config = cifar10_resnet18_nobias_hyperparams.config
        elif args.model_name == 'resnet18_nobias_nobn':
            config = cifar10_resnet18_nobias_nobn_hyperparams.config
        else:
            raise NotImplementedError

    if args.second_model_name is not None:
        if 'vgg' in args.second_model_name:
            if 'half' in args.second_model_name:
                second_config = cifar10_vgg_hyperparams_half.config
            elif 'doub' in args.second_model_name:
                second_config = cifar10_vgg_hyperparams_doub.config
            elif 'quad' in args.second_model_name:
                second_config = cifar10_vgg_hyperparams_quad.config
            elif args.second_model_name == 'vgg11_nobias':
                second_config = cifar10_vgg_hyperparams.config
            else:
                raise NotImplementedError
        elif 'resnet' in args.second_model_name:
            if args.second_model_name == 'resnet18_nobias':
                second_config= cifar10_resnet18_nobias_hyperparams.config
            elif args.second_model_name == 'resnet18_nobias_nobn':
                config = cifar10_resnet18_nobias_nobn_hyperparams.config
            else:
                raise  NotImplementedError
    else:
        second_config = config

    return config, second_config

def get_model_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

