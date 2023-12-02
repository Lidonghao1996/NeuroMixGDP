# Code for NeuroMixGDP
## Environment
This repo contains code for NeuroMixGDP. It is heavily based on the HandcraftDP repo. To run the code, first install the required packages. The training code is based on **pytorch**, but to extract the feature you need to install **Kymatio** and **Pytorch<=1.7** for ScaateringNet and **TenserFlow** for Simclr. We give a sample of required packages in requirement.txt.

```
conda create -n NeuroMixGDP python=3.8
conda activate NeuroMixGDP
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
pip install tensorflow==2.7
pip install kymatio
pip install opacus==0.13
pip install scikit-learn
pip install tensorflow_hub
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```

## Reproduce our results
To reproduce our results, one needs to first extract features and then use the stored features to conduct NeuroMixGDP and train the classifier. 

For NeuroMixGDP on MNIST dataset:
```
python -m transfer.extract_scattering --dataset mnist # extract features and store them
m=64
C=1
lambda=1
eps=1
python3 -m transfer.exp_scatteringnet --feature_path=transfer/features/_scattering_mnist --batch_size=256 --lr=0.001 --optim Adam --noise_multiplier=0 --m $m --eps $eps --T 60000 --Cx $C  --Cy $C --lamb $lambda
```

For NeuroMixGDP on CIFAR10 dataset:
```
python -m transfer.extract_simclr_cifar10 # extract features and store them
m=64
C=1
lambda=1
eps=1
python3 -m transfer.exp_simclr --feature_path=transfer/features/cifar10_simclr_r152_3x_sk1 --batch_size=256 --lr=0.001 --optim Adam --noise_multiplier=0 --m $m --eps $eps --T 50000  --Cx $C --Cy $C --dataset cifar10 --lamb $lambda 
````

For NeuroMixGDP on CIFAR100 dataset:
```
python -m transfer.extract_simclr_cifar100 # extract features and store them
m=64
C=1
lambda=1
eps=1
python3 -m transfer.exp_simclr --feature_path=transfer/features/cifar100_simclr_r152_3x_sk1 --batch_size=256 --lr=0.001 --optim Adam --noise_multiplier=0 --m $m --eps $eps --T 50000  --Cx $C --Cy $C --dataset cifar100 --lamb $lambda 
````

For NeuroMixGDP-HS on CIFAR100 dataset:
```
m=64
C=1
lambda=1
eps=1
classes=3 # the expected class in the mixed sampled
python3 -m transfer.exp_simclr_hs --feature_path=transfer/features/cifar100_simclr_r152_3x_sk1 --batch_size=256 --lr=0.001 --optim Adam --noise_multiplier=0 --m $m --eps $eps --T 50000  --Cx $C --Cy $C --dataset cifar100 --lamb $lambda --classes $classes
````

