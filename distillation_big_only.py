import torch
import numpy as np
import parameters
import utils
from data import get_dataloader
import routines
import baseline
import wasserstein_ensemble
import os
import sys
from model import get_model_from_name
import copy
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

def recheck_accuracy(args, models, test_loader):
    # Additional flag of recheck_acc to supplement the legacy flag recheck_cifar
    if args.recheck_cifar or args.recheck_acc:
        recheck_accuracies = []
        for model in models:
            log_dict = {}
            log_dict['test_losses'] = []
            recheck_accuracies.append(routines.test(args, model, test_loader, log_dict))
        print("Rechecked accuracies are ", recheck_accuracies)


def get_dataloaders(args, config):
    if args.dataset == 'mnist':
        train_loader, test_loader = get_dataloader(args)
        retrain_loader, _ = get_dataloader(args, no_randomness=args.no_random_trainloaders)
    elif args.dataset.lower()[0:7] == 'cifar10':
        assert config is not None
        args.cifar_init_lr = config['optimizer_learning_rate']
        if args.second_model_name is not None:
            assert second_config is not None
            assert args.cifar_init_lr == second_config['optimizer_learning_rate']
            # also the below things should be fine as it is just dataloader loading!
        print('loading {} dataloaders'.format(args.dataset.lower()))
        train_loader, test_loader = cifar_train.get_dataset(config)
        retrain_loader, _ = cifar_train.get_dataset(config, no_randomness=args.no_random_trainloaders)

    return train_loader, test_loader, retrain_loader


def load_pretrained_models(args, config, second_config=None):
    print("------- Loading pre-trained models -------")
    ensemble_experiment = args.load_models.split('/')
    if len(ensemble_experiment) > 1:
        # both the path and name of the experiment have been specified
        ensemble_dir = args.load_models
    elif len(ensemble_experiment) == 1:
        # otherwise append the directory before!
        ensemble_root_dir = "{}/{}_models/".format(args.baseroot, (args.dataset).lower())
        ensemble_dir = ensemble_root_dir + args.load_models

    models = []
    accuracies = []

    for idx in range(args.num_models):
        print("loading model with idx {} and checkpoint_type is {}".format(idx, args.ckpt_type))

        if args.dataset.lower()[0:7] == 'cifar10' and (
                args.model_name.lower()[0:5] == 'vgg11' or args.model_name.lower()[0:6] == 'resnet'):
            if idx == 0:
                config_used = config
            elif idx == 1:
                config_used = second_config

            model, accuracy = cifar_train.get_pretrained_model(
                config_used, os.path.join(ensemble_dir, 'model_{}/{}.checkpoint'.format(idx, args.ckpt_type)),
                args.gpu_id, relu_inplace=not args.prelu_acts  # if you want pre-relu acts, set relu_inplace to False
            )
        else:
            model, accuracy = routines.get_pretrained_model(
                args, os.path.join(ensemble_dir, 'model_{}/{}.checkpoint'.format(idx, args.ckpt_type)), idx=idx
            )

        models.append(model)

        accuracies.append(accuracy)
    print("Done loading all the models")

    return models, accuracies

def test_model(args, model, test_loader):
    log_dict = {}
    log_dict['test_losses'] = []
    return routines.test(args, model, test_loader, log_dict)

def loss_fn_kd(outputs, labels, teacher_outputs, params):
    # Source: https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params.alpha
    T = params.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


def distillation(args, teachers, student, train_loader, test_loader, device):
    # Inspiration: https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/evaluate.py

    for teacher in teachers:
        teacher.eval()

    optimizer = optim.SGD(student.parameters(), lr=args.learning_rate,
                          momentum=args.momentum)

    log_dict = {}
    log_dict['train_losses'] = []
    log_dict['train_counter'] = []
    log_dict['test_losses'] = []

    accuracies = []
    accuracies.append(routines.test(args, student, test_loader, log_dict))
    for epoch_idx in range(0, args.dist_epochs):
        student.train()

        for batch_idx, (data_batch, labels_batch) in enumerate(train_loader):

            # move to GPU if available
            if args.gpu_id != -1:
                data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)

            # compute mean teacher output
            teacher_outputs = []
            for teacher in teachers:
                teacher_outputs.append(teacher(data_batch, disable_logits=True))
            teacher_outputs = torch.stack(teacher_outputs)
            teacher_outputs = teacher_outputs.mean(dim=0)
            optimizer.zero_grad()
            # get student output
            student_output = student(data_batch, disable_logits=True)

            # knowledge distillation loss
            loss = loss_fn_kd(student_output, labels_batch, teacher_outputs, args)
            loss.backward()
            # update student
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_idx, batch_idx * len(data_batch), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))
                log_dict['train_losses'].append(loss.item())
                log_dict['train_counter'].append((batch_idx * 64) + ((epoch_idx - 1) * len(train_loader.dataset)))

        accuracies.append(routines.test(args, student, test_loader, log_dict))

    return student, accuracies

if __name__ == '__main__':

    NUMPY_SEED = 100
    TORCH_SEED = 100
    torch.manual_seed(TORCH_SEED)
    np.random.seed(NUMPY_SEED)

    print("------- Setting up parameters -------")
    args = parameters.get_parameters()

    if args.width_ratio != 1:
        if not args.proper_marginals:
            print('setting proper marginals to True (needed for width_ratio!=1 mode)')
            args.proper_marginals = True
        if args.eval_aligned:
            print('setting eval aligned to False (needed for width_ratio!=1 mode)')
            args.eval_aligned = False

    print("The parameters are: \n", args)


    config, second_config = utils._get_config(args)

    setattr(args, 'autoencoder', False)
    train_loader, test_loader, retrain_loader = get_dataloaders(args, config)

    models, accuracies = load_pretrained_models(args, config)

    recheck_accuracy(args, models, test_loader)

    for idx, model in enumerate(models):
        print(f'model {idx} size is ', utils.get_model_size(model))
        test_model(args, model, test_loader)

    if args.gpu_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))

    print("------- Prediction based ensembling -------")
    prediction_acc = baseline.prediction_ensembling(args, models, test_loader)


    print("------- Geometric Ensembling -------")
    activations = utils.get_model_activations(args, models, config=config)
    geometric_acc, geometric_model = wasserstein_ensemble.geometric_ensembling_modularized(args, models, train_loader,
                                                                                           test_loader, activations)
    utils.get_model_size(geometric_model)

    print("------- Distillation!! -------")
    distilled_model = get_model_from_name(args, idx=1)
    distilled_model = distilled_model.to(device)
    utils.get_model_size(distilled_model)

    distill_scratch_init_acc = test_model(args, distilled_model, test_loader)

    distillation_results = {}

    print("------- Distilling Big to scratch -------")
    _, acc = distillation(args, [models[0]], copy.deepcopy(distilled_model), train_loader, test_loader, device)
    distillation_results['scratch_distill_from_big'] = acc

    print("------- Distilling Big to OT Avg. -------")
    _, acc = distillation(args, [models[0]], copy.deepcopy(geometric_model), train_loader, test_loader, device)
    distillation_results['geometric_distill_from_big'] = acc

    print("------- Distilling Big to Model B -------")
    _, acc = distillation(args, [models[0]], copy.deepcopy(models[1]), train_loader, test_loader, device)
    distillation_results['model_b_distill_from_big'] = acc


    if args.save_result_file != '':

        results_dic = {}
        results_dic['exp_name'] = args.exp_name

        for idx, acc in enumerate(accuracies):
            results_dic['model{}_acc'.format(idx)] = acc

        results_dic['geometric_acc'] = geometric_acc
        results_dic['prediction_acc'] = prediction_acc
        results_dic['distill_scratch_init_acc'] = distill_scratch_init_acc

        # distillation acc results
        for distill_name, acc in distillation_results.items():
            results_dic[f'best_{distill_name}'] = max(acc)

        for distill_name, acc in distillation_results.items():
            results_dic[f'idx_{distill_name}'] = np.argmax(np.array(acc))

        for distill_name, acc in distillation_results.items():
            results_dic[f'acc_{distill_name}'] = acc

        utils.save_results_params_csv(
            os.path.join(args.csv_dir, args.save_result_file),
            results_dic,
            args
        )

        print('----- Saved results at {} ------'.format(args.save_result_file))
        print(results_dic)


    print("FYI: the parameters were: \n", args)

    print("------- ------- ------- ------- -------")
    