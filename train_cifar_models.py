import sys
import utils as myutils
PATH_TO_CIFAR = "./cifar/"
sys.path.append(PATH_TO_CIFAR)
import train as cifar_train
import hyperparameters.vgg11_cifar10_baseline as cifar10_vgg_hyperparams
import hyperparameters.vgg11_half_cifar10_baseline as cifar10_vgg_half_hyperparams
import hyperparameters.vgg11_doub_cifar10_baseline as cifar10_vgg_doub_hyperparams
import hyperparameters.vgg11_quad_cifar10_baseline as cifar10_vgg_quad_hyperparams
import hyperparameters.resnet18_nobias_cifar10_baseline as cifar10_resnet18_nobias_hyperparams
import hyperparameters.resnet18_nobias_nobn_cifar10_baseline as cifar10_resnet18_nobias_nobn_hyperparams
import copy

# num_models = 10
num_models = 2
# gpus = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
gpus = [6, 6]

def main():
    if len(sys.argv) >=2:
        model_type = str(sys.argv[1])
        if '@' in model_type:
            model_type, architecture_type = model_type.split('@')
        else:
            architecture_type = 'vgg'
    else:
        model_type = 'cifar10'
        architecture_type = 'vgg'

    if len(sys.argv) >=3:
        sub_type = str(sys.argv[2]) + '_'
        sub_type_str = str(sys.argv[2])
    else:
        sub_type = ''
        sub_type_str = 'plain'

    if len(sys.argv) >= 4:
        gpu_num = int(sys.argv[3])
        gpus = [gpu_num] * num_models

    base_config = globals()[f'{model_type}_{architecture_type}_{sub_type}hyperparams'].config
    print('base_config is ', base_config)
    print("gpus are ", gpus)
    print(f'Model type is {model_type} and sub_type is {sub_type_str}')

    timestamp = myutils.get_timestamp_other()

    assert len(gpus) == num_models
    for idx in range(num_models):
        model_config = copy.deepcopy(base_config)
        model_config['seed'] = model_config['seed'] + idx
        print("Model with idx {} runnning with seed {} on GPU {}".format(idx, model_config['seed'], gpus[idx]))

        model_output_dir = './cifar_models/exp_{}_{}_{}/model_{}/'.format(model_type, sub_type_str, timestamp, idx)
        print("This model with idx {} will be saved at {}".format(idx, model_output_dir))

        accuracy = cifar_train.main(model_config, model_output_dir, gpus[idx])
        print("The accuracy of model with idx {} is {}".format(idx, accuracy))

    print("Done training all the models")

if __name__ == '__main__':
    main()