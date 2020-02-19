import torch.nn as nn
import torch.nn.functional as F
import sys
PATH_TO_CIFAR = "./cifar/"
sys.path.append(PATH_TO_CIFAR)
import train as cifar_train

def get_model_from_name(args, idx=-1):
    if idx != -1 and idx == (args.num_models - 1):
        # only passes for the second model
        width_ratio = args.width_ratio
    else:
        width_ratio = -1

    if args.model_name == 'net':
        return Net(args)
    elif args.model_name == 'simplenet':
        return SimpleNet(args)
    elif args.model_name == 'smallmlpnet':
        return SmallMlpNet(args)
    elif args.model_name == 'mlpnet':
        return MlpNet(args, width_ratio=width_ratio)
    elif args.model_name == 'bigmlpnet':
        return BigMlpNet(args)
    elif args.model_name == 'cifarmlpnet':
        return CifarMlpNet(args)
    elif args.model_name[0:3] == 'vgg' or args.model_name[0:3] == 'res':
        if args.second_model_name is None or idx == 0:
            barebone_config = {'model': args.model_name, 'dataset': args.dataset}
        else:
            barebone_config = {'model': args.second_model_name, 'dataset': args.dataset}

        # if you want pre-relu acts, set relu_inplace to False
        return cifar_train.get_model(barebone_config, args.gpu_id, relu_inplace=not args.prelu_acts)
        

class LogisticRegressionModel(nn.Module):
    # default input and output dim for
    def __init__(self, input_dim=784, output_dim=10):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        out = F.softmax(self.linear(x))
        return out

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, bias= not args.disable_bias)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, bias= not args.disable_bias)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50, bias= not args.disable_bias)
        self.fc2 = nn.Linear(50, 10, bias= not args.disable_bias)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class SimpleNet(nn.Module):
    def __init__(self, args):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, args.num_hidden_nodes, bias= not args.disable_bias)
        self.fc2 = nn.Linear(args.num_hidden_nodes, 10, bias= not args.disable_bias)
        self.enable_dropout = args.enable_dropout

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class MlpNet(nn.Module):
    def __init__(self, args, width_ratio=-1):
        super(MlpNet, self).__init__()
        if args.dataset == 'mnist':
            # 28 x 28 x 1
            input_dim = 784
        elif args.dataset.lower() == 'cifar10':
            # 32 x 32 x 3
            input_dim = 3072
        if width_ratio != -1:
            self.width_ratio = width_ratio
        else:
            self.width_ratio = 1

        self.fc1 = nn.Linear(input_dim, int(args.num_hidden_nodes1/self.width_ratio), bias=not args.disable_bias)
        self.fc2 = nn.Linear(int(args.num_hidden_nodes1/self.width_ratio), int(args.num_hidden_nodes2/self.width_ratio), bias=not args.disable_bias)
        self.fc3 = nn.Linear(int(args.num_hidden_nodes2/self.width_ratio), int(args.num_hidden_nodes3/self.width_ratio), bias=not args.disable_bias)
        self.fc4 = nn.Linear(int(args.num_hidden_nodes3/self.width_ratio), 10, bias=not args.disable_bias)
        self.enable_dropout = args.enable_dropout

    def forward(self, x, disable_logits=False):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc4(x)

        if disable_logits:
            return x
        else:
            return F.log_softmax(x)


class SmallMlpNet(nn.Module):
    def __init__(self, args):
        super(SmallMlpNet, self).__init__()
        self.fc1 = nn.Linear(784, args.num_hidden_nodes1, bias=not args.disable_bias)
        self.fc2 = nn.Linear(args.num_hidden_nodes1, args.num_hidden_nodes2, bias=not args.disable_bias)
        self.fc3 = nn.Linear(args.num_hidden_nodes2, 10, bias=not args.disable_bias)
        self.enable_dropout = args.enable_dropout

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc3(x)

        return F.log_softmax(x)


class BigMlpNet(nn.Module):
    def __init__(self, args):
        super(BigMlpNet, self).__init__()
        if args.dataset == 'mnist':
            # 28 x 28 x 1
            input_dim = 784
        elif args.dataset.lower() == 'cifar10':
            # 32 x 32 x 3
            input_dim = 3072
        self.fc1 = nn.Linear(input_dim, args.num_hidden_nodes1, bias=not args.disable_bias)
        self.fc2 = nn.Linear(args.num_hidden_nodes1, args.num_hidden_nodes2, bias=not args.disable_bias)
        self.fc3 = nn.Linear(args.num_hidden_nodes2, args.num_hidden_nodes3, bias=not args.disable_bias)
        self.fc4 = nn.Linear(args.num_hidden_nodes3, args.num_hidden_nodes4, bias=not args.disable_bias)
        self.fc5 = nn.Linear(args.num_hidden_nodes4, 10, bias=not args.disable_bias)
        self.enable_dropout = args.enable_dropout

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc4(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc5(x)

        return F.log_softmax(x)


class CifarMlpNet(nn.Module):
    def __init__(self, args):
        super(CifarMlpNet, self).__init__()
        input_dim = 3072
        self.fc1 = nn.Linear(input_dim, 1024, bias=not args.disable_bias)
        self.fc2 = nn.Linear(1024, 512, bias=not args.disable_bias)
        self.fc3 = nn.Linear(512, 128, bias=not args.disable_bias)
        self.fc4 = nn.Linear(128, 10, bias=not args.disable_bias)
        self.enable_dropout = args.enable_dropout

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc4(x)

        return F.log_softmax(x)
