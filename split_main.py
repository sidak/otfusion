import parameters
from data import get_dataloader
import torch
import routines
import baseline
import wasserstein_ensemble
import os
import utils
import numpy as np
import sys

import partition

PATH_TO_CIFAR = "./cifar/"
sys.path.append(PATH_TO_CIFAR)
import train as cifar_train
from tensorboardX import SummaryWriter


if __name__ == '__main__':

    NUMPY_SEED = 100
    TORCH_SEED = 100
    torch.manual_seed(TORCH_SEED)
    np.random.seed(NUMPY_SEED)

    print("------- Setting up parameters -------")
    args = parameters.get_parameters()

    if args.width_ratio !=1:
        if not args.proper_marginals:
            print('setting proper marginals to True (needed for width_ratio!=1 mode)')
            args.proper_marginals = True
        if args.eval_aligned:
            print('setting eval aligned to False (needed for width_ratio!=1 mode)')
            args.eval_aligned = False

    print("The parameters are: \n", args)

    # loading configuration
    config, second_config = utils._get_config(args)
    args.config = config
    args.second_config = second_config

    # obtain trained models
    if args.load_models != '':
        print("------- Loading pre-trained models -------")

        # currently mnist is not supported!
        # assert args.dataset != 'mnist'

        # ensemble_experiment = "exp_2019-04-23_18-08-48/"
        # ensemble_experiment = "exp_2019-04-24_02-20-26"

        ensemble_experiment = args.load_models.split('/')
        if len(ensemble_experiment) > 1:
            # both the path and name of the experiment have been specified
            ensemble_dir = args.load_models
        elif len(ensemble_experiment) == 1:
            # otherwise append the directory before!
            ensemble_root_dir = "{}/{}_models/".format(args.baseroot, (args.dataset).lower())
            ensemble_dir = ensemble_root_dir + args.load_models

        utils.mkdir(ensemble_dir)
        # checkpoint_type = 'final'  # which checkpoint to use for ensembling (either of 'best' or 'final)

        if args.dataset=='mnist':
            train_loader, test_loader = get_dataloader(args)
        elif args.dataset.lower() == 'cifar10':
            args.cifar_init_lr = config['optimizer_learning_rate']
            if args.second_model_name is not None:
                assert second_config is not None
                assert args.cifar_init_lr == second_config['optimizer_learning_rate']
                # also the below things should be fine as it is just dataloader loading!
            print('loading {} dataloaders'.format(args.dataset.lower()))
            train_loader, test_loader = cifar_train.get_dataset(config)

        models = []
        accuracies = []
        local_accuracies = []
        for idx in range(args.num_models):
            print("loading model with idx {} and checkpoint_type is {}".format(idx, args.ckpt_type))

            if args.dataset.lower()[0:7] == 'cifar10' and (args.model_name.lower()[0:5] == 'vgg11' or args.model_name.lower()[0:6] == 'resnet'):
                if idx == 0:
                    config_used = config
                elif idx == 1:
                    config_used = second_config

                model, accuracy = cifar_train.get_pretrained_model(
                        config_used, os.path.join(ensemble_dir, 'model_{}/{}.checkpoint'.format(idx, args.ckpt_type)),
                        args.gpu_id, relu_inplace=not args.prelu_acts # if you want pre-relu acts, set relu_inplace to False
                )
            else:
                model, accuracy, local_accuracy = routines.get_pretrained_model(
                        args, os.path.join(ensemble_dir, 'model_{}/{}.checkpoint'.format(idx, args.ckpt_type)), data_separated=True, idx = idx
                )
            models.append(model)
            accuracies.append(accuracy)
            local_accuracies.append(local_accuracy)
        print("Done loading all the models")

        # Additional flag of recheck_acc to supplement the legacy flag recheck_cifar
        if args.recheck_cifar or args.recheck_acc:
            recheck_accuracies = []
            for model in models:
                log_dict = {}
                log_dict['test_losses'] = []
                recheck_accuracies.append(routines.test(args, model, test_loader, log_dict))
            print("Rechecked accuracies are ", recheck_accuracies)

        # print('checking named modules of model0 for use in compute_activations!', list(models[0].named_modules()))

    else:
        # get dataloaders
        print("------- Obtain dataloaders -------")
        train_loader, test_loader = get_dataloader(args)

        if args.partition_type == 'labels':
            print("------- Split dataloaders by labels -------")
            choice = [int(x) for x in args.choice.split()]
            (trailo_a, teslo_a), (trailo_b, teslo_b), other = partition.split_mnist_by_labels(args, train_loader, test_loader, choice=choice)
            print("------- Training independent models -------")
            models, accuracies, local_accuracies = routines.train_data_separated_models(args, [trailo_a, trailo_b],
                                              [teslo_a, teslo_b], test_loader, [choice, list(other)])
        elif args.partition_type == 'personalized':
            assert args.dataset == 'mnist'
            print("------- Split dataloaders wrt personalized data setting-------")
            trailo_a, trailo_b, personal_trainset, other_trainset = partition.get_personalized_split(args, personal_label = args.personal_class_idx,
                                                                  split_frac= args.personal_split_frac, is_train=True, return_dataset=True)
            teslo_a, teslo_b, personal_testset, other_testset = partition.get_personalized_split(args, personal_label=args.personal_class_idx,
                                                                  split_frac=args.personal_split_frac, is_train=False, return_dataset=True)
            print("------- Training independent models -------")

            other = list(range(0, 10))
            other.remove(args.personal_class_idx)
            models, accuracies, local_accuracies = routines.train_data_separated_models(args, [trailo_a, trailo_b],
                                                                [teslo_a, teslo_b], test_loader,
                                                                [list(range(0, 10)), other])
        elif args.partition_type == 'small_big':
            assert args.dataset == 'mnist'
            print("------- Split dataloaders wrt small big data setting-------")
            trailo_a, trailo_b, personal_trainset, other_trainset = partition.get_small_big_split(args,
                                                      split_frac= args.personal_split_frac, is_train=True, return_dataset=True)
            teslo_a, teslo_b, personal_testset, other_testset = partition.get_small_big_split(args,
                                                      split_frac=args.personal_split_frac, is_train=False, return_dataset=True)
            print("------- Training independent models -------")

            choices = list(range(0, 10))
            models, accuracies, local_accuracies = routines.train_data_separated_models(args, [trailo_a, trailo_b],
                                                                [teslo_a, teslo_b], test_loader,
                                                                [choices, choices])

    for idx, model in enumerate(models):
        setattr(args, f'params_model_{idx}', utils.get_model_size(model))

    personal_dataset = None
    if args.partition_type == 'personalized' or args.partition_type == 'small_big':
        if args.partition_dataloader == 0:
            personal_dataset = personal_trainset
        elif args.partition_dataloader == 1:
            personal_dataset = other_trainset

    activations = utils.get_model_activations(args, models, config=config, personal_dataset=personal_dataset)

    # run geometric aka wasserstein ensembling
    print("------- Geometric Ensembling -------")

    geometric_acc, geometric_model = wasserstein_ensemble.geometric_ensembling_modularized(args, models, train_loader, test_loader, activations)

    args.params_geometric = utils.get_model_size(geometric_model)

    # run baselines
    print("------- Prediction based ensembling -------")
    prediction_acc = baseline.prediction_ensembling(args, models, test_loader)

    print("------- Naive ensembling of weights -------")
    naive_acc, naive_model = baseline.naive_ensembling(args, models, test_loader)

    if args.retrain > 0:
        print('-------- Retraining the models ---------')
        if args.tensorboard:
            tensorboard_dir = os.path.join(args.tensorboard_root, args.exp_name)
            utils.mkdir(tensorboard_dir)
            print("Tensorboard experiment directory: {}".format(tensorboard_dir))
            tensorboard_obj = SummaryWriter(log_dir=tensorboard_dir)
        else:
            tensorboard_obj = None

        if args.retrain_avg_only:
            initial_acc = [geometric_acc, naive_acc]
            _, best_retrain_acc = routines.retrain_models(args, [geometric_model, naive_model], train_loader, test_loader, config, tensorboard_obj=tensorboard_obj, initial_acc=initial_acc)
            args.retrain_geometric_best = best_retrain_acc[0]
            args.retrain_naive_best = best_retrain_acc[1]

        else:
            initial_acc = [*accuracies, geometric_acc, naive_acc]
            _, best_retrain_acc = routines.retrain_models(args, [*models, geometric_model, naive_model], train_loader, test_loader, config, tensorboard_obj=tensorboard_obj, initial_acc=initial_acc)
            args.retrain_model0_best = best_retrain_acc[0]
            args.retrain_model1_best = best_retrain_acc[1]
            args.retrain_geometric_best = best_retrain_acc[2]
            args.retrain_naive_best = best_retrain_acc[3]

    if args.save_result_file != '':

        results_dic = {}
        results_dic['exp_name'] = args.exp_name

        for idx, acc in enumerate(accuracies):
            results_dic['model{}_acc'.format(idx)] = acc

        for idx, local_acc in enumerate(local_accuracies):
            results_dic['model{}_local_acc'.format(idx)] = local_acc


        results_dic['geometric_acc'] = geometric_acc
        results_dic['prediction_acc'] = prediction_acc
        results_dic['naive_acc'] = naive_acc

        # Additional statistics
        results_dic['geometric_gain'] = geometric_acc - max(accuracies)
        results_dic['geometric_gain_%'] = ((geometric_acc - max(accuracies))*100.0)/max(accuracies)
        results_dic['prediction_gain'] = prediction_acc - max(accuracies)
        results_dic['prediction_gain_%'] = ((prediction_acc - max(accuracies)) * 100.0) / max(accuracies)
        results_dic['relative_loss_wrt_prediction'] = results_dic['prediction_gain_%'] - results_dic['geometric_gain_%']

        if args.eval_aligned:
            results_dic['model0_aligned'] = args.model0_aligned_acc
            
        # Save retrain statistics!
        if args.retrain > 0:
            results_dic['retrain_geometric_best'] = args.retrain_geometric_best * 100
            results_dic['retrain_naive_best'] = args.retrain_naive_best * 100
            if not args.retrain_avg_only:
                results_dic['retrain_model0_best'] = args.retrain_model0_best * 100
                results_dic['retrain_model1_best'] = args.retrain_model1_best * 100
            results_dic['retrain_epochs'] = args.retrain

        utils.save_results_params_csv(
            os.path.join(args.csv_dir, args.save_result_file),
            results_dic,
            args
        )

        print('----- Saved results at {} ------'.format(args.save_result_file))
        print(results_dic)

    
    print("FYI: the parameters were: \n", args)

