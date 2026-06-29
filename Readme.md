<div align="center">

<h1>	
Intrinsically Stable Spiking Neural Networks: Overcoming the Performance Barrier in the Absence of Batch Normalization</h1>

</div>

## 🚀 Introduction

This is the official PyTorch implementation of the paper **Intrinsically Stable Spiking Neural Networks: Overcoming the Performance Barrier in the Absence of Batch Normalization**, accepted at **ECCV 2026**.


## Requirements
- python==3.10
- numpy==1.23.5
- spikingjelly==0.0.0.0.14
- torch==2.2.0
- torchvision=0.17.0
- timm=1.0.7

## Usage
Training on the CIFAR-10 dataset with AMP and Mixup/Cutmix: 
``` bash
conda activate xxx
CUDA_VISIBLE_DEVICES=0 \
python train.py --batch_size 128 --dataset_path '/ssd/Datasets/CIFAR10/' --dataset 'cifar10'\
                --class_number 10 --epochs 256 --lr 0.02 --weight_decay 5e-4 --amp\
                --timestep 4 --alpha 0.5 --mixup --workers 4 --print_freq 30 --name 'CIFAR10'
```
Training on the ImageNet dataset with AMP: 
``` bash
conda activate xxx
python train.py --batch_size 256 --dataset_path '/ssd/Datasets/ImageNet/' --dataset 'imagenet'\
                --class_number 1000 --epochs 128 --lr 0.2 --weight_decay 0 --amp\
                --timestep 4 --alpha 0.5 --workers 16 --print_freq 200 --name 'ImageNet'\
                --multiprocessing_distributed
```