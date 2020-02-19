import parameters
from data import get_dataloader
import routines
import baseline
import wasserstein_ensemble
import os
import utils
import numpy as np
import sys
import torch


PATH_TO_CIFAR = "./cifar/"
sys.path.append(PATH_TO_CIFAR)
import train as cifar_train
from tensorboardX import SummaryWriter

if __name__ == '__main__':

    print("------- Setting up parameters -------")
    args = parameters.get_parameters()
    print("The parameters are: \n", args)

    if args.deterministic:
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

        # checkpoint_type = 'final'  # which checkpoint to use for ensembling (either of 'best' or 'final)

        if args.dataset=='mnist':
            train_loader, test_loader = get_dataloader(args)
            retrain_loader, _ = get_dataloader(args, no_randomness=args.no_random_trainloaders)
        elif args.dataset.lower()[0:7] == 'cifar10':
            args.cifar_init_lr = config['optimizer_learning_rate']
            if args.second_model_name is not None:
                assert second_config is not None
                assert args.cifar_init_lr == second_config['optimizer_learning_rate']
                # also the below things should be fine as it is just dataloader loading!
            print('loading {} dataloaders'.format(args.dataset.lower()))
            train_loader, test_loader = cifar_train.get_dataset(config)
            retrain_loader, _ = cifar_train.get_dataset(config, no_randomness=args.no_random_trainloaders)


        models = []
        accuracies = []

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
                model, accuracy = routines.get_pretrained_model(
                        args, os.path.join(ensemble_dir, 'model_{}/{}.checkpoint'.format(idx, args.ckpt_type)), idx = idx
                )

            models.append(model)
            accuracies.append(accuracy)
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

        # print('what about named parameters of model0 for use in compute_activations!', [tupl[0] for tupl in list(models[0].named_parameters())])

    else:
        # get dataloaders
        print("------- Obtain dataloaders -------")
        train_loader, test_loader = get_dataloader(args)
        retrain_loader, _ = get_dataloader(args, no_randomness=args.no_random_trainloaders)

        print("------- Training independent models -------")
        models, accuracies = routines.train_models(args, train_loader, test_loader)

    # if args.debug:
    #     print(list(models[0].parameters()))

    if args.same_model!=-1:
        print("Debugging with same model")
        model, acc = models[args.same_model], accuracies[args.same_model]
        models = [model, model]
        accuracies = [acc, acc]

    for name, param in models[0].named_parameters():
        print(f'layer {name} has #params ', param.numel())

    import time
    # second_config is not needed here as well, since it's just used for the dataloader!
    print("Activation Timer start")
    st_time = time.perf_counter()
    activations = utils.get_model_activations(args, models, config=config)
    end_time = time.perf_counter()
    setattr(args, 'activation_time', end_time - st_time)
    print("Activation Timer ends")

    for idx, model in enumerate(models):
        setattr(args, f'params_model_{idx}', utils.get_model_size(model))

    # if args.ensemble_iter == 1:
    #
    # else:
    #     # else just recompute activations inside the method iteratively
    #     activations = None


    # set seed for numpy based calculations
    NUMPY_SEED = 100
    np.random.seed(NUMPY_SEED)

    # run geometric aka wasserstein ensembling
    print("------- Geometric Ensembling -------")
    # Deprecated: wasserstein_ensemble.geometric_ensembling(models, train_loader, test_loader)


    print("Timer start")
    st_time = time.perf_counter()

    geometric_acc, geometric_model = wasserstein_ensemble.geometric_ensembling_modularized(args, models, train_loader, test_loader, activations)
    
    end_time = time.perf_counter()
    print("Timer ends")
    setattr(args, 'geometric_time', end_time - st_time)
    args.params_geometric = utils.get_model_size(geometric_model)

    print("Time taken for geometric ensembling is {} seconds".format(str(end_time - st_time)))
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

        if args.retrain_avg_only or args.retrain_geometric_only:
            if args.retrain_geometric_only:
                initial_acc = [geometric_acc]
                nicks = ['geometric']
                _, best_retrain_acc = routines.retrain_models(args, [geometric_model], retrain_loader,
                                                              test_loader, config, tensorboard_obj=tensorboard_obj,
                                                              initial_acc=initial_acc, nicks=nicks)
                args.retrain_geometric_best = best_retrain_acc[0]
                args.retrain_naive_best = -1
            else:
                if naive_acc < 0:
                    initial_acc = [geometric_acc]
                    nicks = ['geometric']
                    _, best_retrain_acc = routines.retrain_models(args, [geometric_model],
                                                                  retrain_loader, test_loader, config,
                                                                  tensorboard_obj=tensorboard_obj,
                                                                  initial_acc=initial_acc, nicks=nicks)
                    args.retrain_geometric_best = best_retrain_acc[0]
                    args.retrain_naive_best = -1
                else:
                    nicks = ['geometric', 'naive_averaging']
                    initial_acc = [geometric_acc, naive_acc]
                    _, best_retrain_acc = routines.retrain_models(args, [geometric_model, naive_model], retrain_loader, test_loader, config, tensorboard_obj=tensorboard_obj, initial_acc=initial_acc, nicks=nicks)
                    args.retrain_geometric_best = best_retrain_acc[0]
                    args.retrain_naive_best = best_retrain_acc[1]

            args.retrain_model0_best = -1
            args.retrain_model1_best = -1

        else:

            if args.skip_retrain == 0:
                original_models = [models[1]]
                original_nicks = ['model_1']
                original_accuracies = [accuracies[1]]
            elif args.skip_retrain == 1:
                original_models = [models[0]]
                original_nicks = ['model_0']
                original_accuracies = [accuracies[0]]
            elif args.skip_retrain < 0:
                original_models = models
                original_nicks = ['model_0', 'model_1']
                original_accuracies = accuracies
            else:
                raise NotImplementedError

            if naive_acc < 0:
                # this happens in case the two models have different layer sizes
                nicks = original_nicks + ['geometric']
                initial_acc = original_accuracies + [geometric_acc]
                _, best_retrain_acc = routines.retrain_models(args, [*original_models, geometric_model],
                                                              retrain_loader, test_loader, config,
                                                              tensorboard_obj=tensorboard_obj, initial_acc=initial_acc, nicks=nicks)
                args.retrain_naive_best = -1
            else:
                nicks = original_nicks + ['geometric', 'naive_averaging']
                initial_acc = [*original_accuracies, geometric_acc, naive_acc]
                _, best_retrain_acc = routines.retrain_models(args, [*original_models, geometric_model, naive_model], retrain_loader, test_loader, config, tensorboard_obj=tensorboard_obj, initial_acc=initial_acc, nicks=nicks)
                args.retrain_naive_best = best_retrain_acc[len(initial_acc)-1]

            if args.skip_retrain == 0:
                args.retrain_model0_best = -1
                args.retrain_model1_best = best_retrain_acc[0]
            elif args.skip_retrain == 1:
                args.retrain_model0_best = best_retrain_acc[0]
                args.retrain_model1_best = -1
            elif args.skip_retrain < 0:
                args.retrain_model0_best = best_retrain_acc[0]
                args.retrain_model1_best = best_retrain_acc[1]

            args.retrain_geometric_best = best_retrain_acc[len(original_models)]

    if args.save_result_file != '':

        results_dic = {}
        results_dic['exp_name'] = args.exp_name

        for idx, acc in enumerate(accuracies):
            results_dic['model{}_acc'.format(idx)] = acc

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

        results_dic['geometric_time'] = args.geometric_time
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
