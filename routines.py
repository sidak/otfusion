import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import get_model_from_name
from data import get_dataloader
import sys
PATH_TO_CIFAR = "./cifar/"
sys.path.append(PATH_TO_CIFAR)
import train as cifar_train
import copy

def get_trained_model(args, id, random_seed, train_loader, test_loader):
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    network = get_model_from_name(args, idx=id)

    optimizer = optim.SGD(network.parameters(), lr=args.learning_rate,
                          momentum=args.momentum)
    if args.gpu_id!=-1:
        network = network.cuda(args.gpu_id)
    log_dict = {}
    log_dict['train_losses'] = []
    log_dict['train_counter'] = []
    log_dict['test_losses'] = []
    # log_dict['test_counter'] = [i * len(test_loader.dataset) for i in range(args.n_epochs + 1)]
    # print(list(network.parameters()))
    acc = test(args, network, test_loader, log_dict)
    for epoch in range(1, args.n_epochs + 1):
        train(args, network, optimizer, train_loader, log_dict, epoch, model_id=str(id))
        acc = test(args, network, test_loader, log_dict)
    return network, acc

def check_freezed_params(model, frozen):
    flag = True
    for idx, param in enumerate(model.parameters()):
        if idx >= len(frozen):
            return flag

        flag = flag and (param.data == frozen[idx].data).all()

    return flag

def get_intmd_retrain_model(args, random_seed, network, aligned_wts, train_loader, test_loader):
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    num_params_aligned = len(aligned_wts)
    for idx, param in enumerate(network.parameters()):
        if idx < num_params_aligned:
            param.requires_grad = False

    print("number of layers that are intmd retrained ", len(list(network.parameters()))-num_params_aligned)
    optimizer = optim.SGD(network.parameters(), lr=args.learning_rate * args.intmd_retrain_lrdec,
                          momentum=args.momentum)
    log_dict = {}
    log_dict['train_losses'] = []
    log_dict['train_counter'] = []
    log_dict['test_losses'] = []
    # log_dict['test_counter'] = [i * len(test_loader.dataset) for i in range(args.n_epochs + 1)]
    # print(list(network.parameters()))
    acc = test(args, network, test_loader, log_dict)
    for epoch in range(1, args.intmd_retrain_epochs + 1):
        train(args, network, optimizer, train_loader, log_dict, epoch, model_id=str(id))
        acc = test(args, network, test_loader, log_dict)

    print("Finally accuracy of model {} after intermediate retraining for {} epochs with lr decay {} is {}".format(
        random_seed, args.intmd_retrain_epochs, args.intmd_retrain_lrdec, acc
    ))

    assert check_freezed_params(network, aligned_wts)
    return network

def get_trained_data_separated_model(args, id, local_train_loader, local_test_loader, test_loader, base_net=None):
    torch.backends.cudnn.enabled = False
    if base_net is not None:
        network = copy.deepcopy(base_net)
    else:
        network = get_model_from_name(args, idx=id)
    optimizer = optim.SGD(network.parameters(), lr=args.learning_rate,
                          momentum=args.momentum)
    if args.gpu_id!=-1:
        network = network.cuda(args.gpu_id)
    log_dict = {}
    log_dict['train_losses'] = []
    log_dict['train_counter'] = []
    log_dict['local_test_losses'] = []
    log_dict['test_losses'] = []
    # log_dict['test_counter'] = [i * len(test_loader.dataset) for i in range(args.n_epochs + 1)]
    # print(list(network.parameters()))
    acc = test(args, network, test_loader, log_dict)
    local_acc = test(args, network, local_test_loader, log_dict, is_local=True)
    for epoch in range(1, args.n_epochs + 1):
        train(args, network, optimizer, local_train_loader, log_dict, epoch, model_id=str(id))
        acc = test(args, network, test_loader, log_dict)
        local_acc = test(args, network, local_test_loader, log_dict, is_local=True)
    return network, acc, local_acc

def get_retrained_model(args, train_loader, test_loader, old_network, tensorboard_obj=None, nick='', start_acc=-1, retrain_seed=-1):
    torch.backends.cudnn.enabled = False
    if args.retrain_lr_decay > 0:
        args.retrain_lr = args.learning_rate / args.retrain_lr_decay
        print('optimizer_learning_rate is ', args.retrain_lr)
    if retrain_seed!=-1:
        torch.manual_seed(retrain_seed)
        
    optimizer = optim.SGD(old_network.parameters(), lr=args.retrain_lr,
                              momentum=args.momentum)
    log_dict = {}
    log_dict['train_losses'] = []
    log_dict['train_counter'] = []
    log_dict['test_losses'] = []
    # log_dict['test_counter'] = [i * len(train_loader.dataset) for i in range(args.n_epochs + 1)]

    acc = test(args, old_network, test_loader, log_dict)
    print("check accuracy once again before retraining starts: ", acc)

    if tensorboard_obj is not None and start_acc != -1:
        tensorboard_obj.add_scalars('test_accuracy_percent/', {nick: start_acc},
                                    global_step=0)
        assert start_acc == acc


    best_acc = -1
    for epoch in range(1, args.retrain + 1):
        train(args, old_network, optimizer, train_loader, log_dict, epoch)
        acc, loss = test(args, old_network, test_loader, log_dict, return_loss=True)

        if tensorboard_obj is not None:
            assert nick != ''
            tensorboard_obj.add_scalars('test_loss/', {nick: loss}, global_step=epoch)
            tensorboard_obj.add_scalars('test_accuracy_percent/', {nick: acc}, global_step=epoch)

        print("At retrain epoch the accuracy is : ", acc)
        best_acc = max(best_acc, acc)

    return old_network, best_acc

def get_pretrained_model(args, path, data_separated=False, idx=-1):
    model = get_model_from_name(args, idx=idx)

    if args.gpu_id != -1:
        state = torch.load(
            path,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cuda:' + str(args.gpu_id))
            ),
        )
    else:
        state = torch.load(
            path,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )


    model_state_dict = state['model_state_dict']

    if 'test_accuracy' not in state:
        state['test_accuracy'] = -1

    if 'epoch' not in state:
        state['epoch'] = -1

    if not data_separated:
        print("Loading model at path {} which had accuracy {} and at epoch {}".format(path, state['test_accuracy'],
                                                                                  state['epoch']))
    else:
        print("Loading model at path {} which had local accuracy {} and overall accuracy {} for choice {} at epoch {}".format(path,
            state['local_test_accuracy'], state['test_accuracy'], state['choice'], state['epoch']))

    model.load_state_dict(model_state_dict)

    if args.gpu_id != -1:
        model = model.cuda(args.gpu_id)

    if not data_separated:
        return model, state['test_accuracy']
    else:
        return model, state['test_accuracy'], state['local_test_accuracy']

def train(args, network, optimizer, train_loader, log_dict, epoch, model_id=-1):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.gpu_id!=-1:
            data = data.cuda(args.gpu_id)
            target = target.cuda(args.gpu_id)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            log_dict['train_losses'].append(loss.item())
            log_dict['train_counter'].append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

            assert args.exp_name == "exp_" + args.timestamp

            os.makedirs('{}/{}'.format(args.result_dir, args.exp_name), exist_ok=True)
            if args.dump_model:
                assert model_id != -1
                torch.save(network.state_dict(), '{}/{}/model_{}_{}.pth'.format(args.result_dir, args.exp_name, args.model_name, model_id))
                torch.save(optimizer.state_dict(), '{}/{}/optimizer_{}_{}.pth'.format(args.result_dir, args.exp_name, args.model_name, model_id))


def test(args, network, test_loader, log_dict, debug=False, return_loss=False, is_local=False):
    network.eval()
    test_loss = 0
    correct = 0
    if is_local:
        print("\n--------- Testing in local mode ---------")
    else:
        print("\n--------- Testing in global mode ---------")

    if args.dataset.lower() == 'cifar10':
        cifar_criterion = torch.nn.CrossEntropyLoss()

    #   with torch.no_grad():
    for data, target in test_loader:
        # print(data.shape, target.shape)
        # if len(target.shape)==1:
        #     data = data.unsqueeze(0)
        #     target = target.unsqueeze(0)
        # print(data, target)
        if args.gpu_id!=-1:
            data = data.cuda(args.gpu_id)
            target = target.cuda(args.gpu_id)

        output = network(data)
        if debug:
            print("output is ", output)

        if args.dataset.lower() == 'cifar10':
            # mnist models return log_softmax outputs, while cifar ones return raw values!
            test_loss += cifar_criterion(output, target).item()
        elif args.dataset.lower() == 'mnist':
            test_loss += F.nll_loss(output, target, size_average=False).item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    print("size of test_loader dataset: ", len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    if is_local:
        string_info = 'local_test'
    else:
        string_info = 'test'
    log_dict['{}_losses'.format(string_info)].append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    ans = (float(correct) * 100.0) / len(test_loader.dataset)

    if not return_loss:
        return ans
    else:
        return ans, test_loss

def train_data_separated_models(args, local_train_loaders, local_test_loaders, test_loader, choices):
    networks = []
    local_accuracies = []
    accuracies = []
    base_nets = []
    base_net = get_model_from_name(args, idx=0)
    base_nets.append(base_net)
    if args.diff_init or args.width_ratio!=1:
        base_nets.append(get_model_from_name(args, idx=1))
    else:
        base_nets.append(base_net)

    for i in range(args.num_models):
        print("\nTraining model {} on its separate data \n ".format(str(i)))
        network, acc, local_acc = get_trained_data_separated_model(args, i,
                                           local_train_loaders[i], local_test_loaders[i], test_loader, base_nets[i])
        networks.append(network)
        accuracies.append(acc)
        local_accuracies.append(local_acc)
        if args.dump_final_models:
            save_final_data_separated_model(args, i, network, local_acc, acc, choices[i])
    return networks, accuracies, local_accuracies


def train_models(args, train_loader, test_loader):
    networks = []
    accuracies = []
    for i in range(args.num_models):
        network, acc = get_trained_model(args, i, i, train_loader, test_loader)
        networks.append(network)
        accuracies.append(acc)
        if args.dump_final_models:
            save_final_model(args, i, network, acc)
    return networks, accuracies

def save_final_data_separated_model(args, idx, model, local_test_accuracy, test_accuracy, choice):
    path = os.path.join(args.result_dir, args.exp_name, 'model_{}'.format(idx))
    os.makedirs(path, exist_ok=True)
    import time
    args.ckpt_type = 'final'
    time.sleep(1)  # workaround for RuntimeError('Unknown Error -1') https://github.com/pytorch/pytorch/issues/10577
    torch.save({
        'args': vars(args),
        'epoch': args.n_epochs,
        'local_test_accuracy': local_test_accuracy,
        'test_accuracy': test_accuracy,
        'choice': str(choice),
        'model_state_dict': model.state_dict(),
    }, os.path.join(path, '{}.checkpoint'.format(args.ckpt_type))
    )


def save_final_model(args, idx, model, test_accuracy):
    path = os.path.join(args.result_dir, args.exp_name, 'model_{}'.format(idx))
    os.makedirs(path, exist_ok=True)
    import time
    args.ckpt_type = 'final'
    time.sleep(1)  # workaround for RuntimeError('Unknown Error -1') https://github.com/pytorch/pytorch/issues/10577
    torch.save({
        'args': vars(args),
        'epoch': args.n_epochs,
        'test_accuracy': test_accuracy,
        'model_state_dict': model.state_dict(),
    }, os.path.join(path, '{}.checkpoint'.format(args.ckpt_type))
    )

def retrain_models(args, old_networks, train_loader, test_loader, config, tensorboard_obj=None, initial_acc=None, nicks=None):
    accuracies = []
    retrained_networks = []
    # nicks = []

    # assert len(old_networks) >= 4

    for i in range(len(old_networks)):
        nick = nicks[i]
        # if i == len(old_networks) - 1:
        #     nick = 'naive_averaging'
        # elif i == len(old_networks) - 2:
        #     nick = 'geometric'
        # else:
        #     nick = 'model_' + str(i)
        # nicks.append(nick)
        print("Retraining model : ", nick)

        if initial_acc is not None:
            start_acc = initial_acc[i]
        else:
            start_acc = -1
        if args.dataset.lower()[0:7] == 'cifar10':

            if args.reinit_trainloaders:
                print('reiniting trainloader')
                retrain_loader, _ = cifar_train.get_dataset(config, no_randomness=args.no_random_trainloaders)
            else:
                retrain_loader = train_loader

            output_root_dir = "{}/{}_models_ensembled/".format(args.baseroot, (args.dataset).lower())
            output_root_dir = os.path.join(output_root_dir, args.exp_name, nick)
            os.makedirs(output_root_dir, exist_ok=True)

            retrained_network, acc = cifar_train.get_retrained_model(args, retrain_loader, test_loader, old_networks[i], config, output_root_dir, tensorboard_obj=tensorboard_obj, nick=nick, start_acc=initial_acc[i])
            
        elif args.dataset.lower() == 'mnist':

            if args.reinit_trainloaders:
                print('reiniting trainloader')
                retrain_loader, _ = get_dataloader(args, no_randomness=args.no_random_trainloaders)
            else:
                retrain_loader = train_loader
                
            start_acc = initial_acc[i]
            retrained_network, acc = get_retrained_model(args, retrain_loader, test_loader, old_network=old_networks[i], tensorboard_obj=tensorboard_obj, nick=nick, start_acc=start_acc, retrain_seed=args.retrain_seed)
        retrained_networks.append(retrained_network)
        accuracies.append(acc)
    return retrained_networks, accuracies


def intmd_retrain_models(args, old_networks, aligned_wts, train_loader, test_loader, config, tensorboard_obj=None, initial_acc=None):
    accuracies = []
    retrained_networks = []
    # nicks = []

    # assert len(old_networks) >= 4

    for i in range(len(old_networks)):

        nick = 'intmd_retrain_model_' + str(i)
        print("Retraining model : ", nick)

        if initial_acc is not None:
            start_acc = initial_acc[i]
        else:
            start_acc = -1
        if args.dataset.lower() == 'cifar10':

            output_root_dir = "{}/{}_models_ensembled/".format(args.baseroot, (args.dataset).lower())
            output_root_dir = os.path.join(output_root_dir, args.exp_name, nick)
            os.makedirs(output_root_dir, exist_ok=True)

            retrained_network, acc = cifar_train.get_retrained_model(args, train_loader, test_loader, old_networks[i], config, output_root_dir, tensorboard_obj=tensorboard_obj, nick=nick, start_acc=start_acc)

        elif args.dataset.lower() == 'mnist':
            # start_acc = initial_acc[i]
            retrained_network, acc = get_intmd_retrain_model(args, train_loader, test_loader, old_network=old_networks[i], tensorboard_obj=tensorboard_obj, nick=nick, start_acc=start_acc)
        retrained_networks.append(retrained_network)
        accuracies.append(acc)
    return retrained_networks, accuracies