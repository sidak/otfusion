### Requirements 

Install the Python Optimal Transport Library

```
pip install POT
```

Other than that, we also need PyTorch v1 or higher and NumPy. (Also, Python 3.6 +)

Before running, unzip the respective pretrained model zip file. Also, **you need to unzip** the `cifar.zip` file for some imports to work.

### Sample commands of one-shot model fusion

#### For MNIST + MLPNet

```
python main.py --gpu-id 1 --model-name mlpnet --n-epochs 10 --save-result-file sample.csv \
--sweep-name exp_sample --exact --correction --ground-metric euclidean --weight-stats \
--activation-histograms --activation-mode raw --geom-ensemble-type acts --sweep-id 21 \
--act-num-samples 200 --ground-metric-normalize none --activation-seed 21 \
--prelu-acts --recheck-acc --load-models ./mnist_models --ckpt-type final \
--past-correction --not-squared --dist-normalize --print-distances --to-download
```

#### For CIFAR10 + VGG11
```
python main.py --gpu-id 1 --model-name vgg11_nobias --n-epochs 300 --save-result-file sample.csv \
--sweep-name exp_sample --correction --ground-metric euclidean --weight-stats \
--geom-ensemble-type wts --ground-metric-normalize none --sweep-id 90 --load-models ./cifar_models/ \
--ckpt-type best --dataset Cifar10 --ground-metric-eff --recheck-cifar --activation-seed 21 \
--prelu-acts --past-correction --not-squared --normalize-wts --exact
```

#### For CIFAR10 + ResNet18

```
python main.py --gpu-id 1 --model-name resnet18_nobias_nobn --n-epochs 300 --save-result-file sample.csv \
--sweep-name exp_sample --exact --correction --ground-metric euclidean --weight-stats \
--activation-histograms --activation-mode raw --geom-ensemble-type acts --sweep-id 21 \
--act-num-samples 200 --ground-metric-normalize none --activation-seed 21 --prelu-acts --recheck-acc \
--load-models ./resnet_models/ --ckpt-type best --past-correction --not-squared  --dataset Cifar10 \
--handle-skips
```

The code and pretrained models correspond to the paper: `Model Fusion via Optimal Transport`. If you use any of the code or pretrained models for your research, please consider citing the paper. 
