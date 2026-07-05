<div align="center">

<h1>	
Intrinsically Stable Spiking Neural Networks: Overcoming the Performance Barrier in the Absence of Batch Normalization</h1>

<!-- Badges -->
[![Paper](https://img.shields.io/badge/Arxiv-2606.31695-B31B1B.svg?style=flat-square)](https://arxiv.org/abs/2606.31695)
[![ECCV 2026](https://img.shields.io/badge/ECCV%202026-Poster-4b44ce.svg?style=flat-square)](https://www.ecva.net/papers.php)
<!-- [![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Paper-4285F4?style=flat-square&logo=google-scholar&logoColor=white)](https://scholar.google.com/scholar?cluster=1814482600796011970) -->

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Paper-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/papers/2606.31695)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/Ruichen0424/IS-SNN)
</div>

## 🚀 Introduction

This is the official PyTorch implementation of the paper **Intrinsically Stable Spiking Neural Networks: Overcoming the Performance Barrier in the Absence of Batch Normalization**, accepted at **ECCV 2026**.

## ✨ Key Highlights

*   **⚡ Zero-Runtime-Overhead SNN:** Removes all dynamic batch normalization (BN) layers. By folding weight standardization operations offline, it restores a pure, hardware-friendly **accumulation-only (addition-only)** inference datapath.
*   **📉 Solves Firing-Rate Decay:** Addresses the fundamental issue of catastrophic firing-rate decay/saturation in BN-free SNNs through theoretical *Topology-Aware Weight Standardization* and *Modified Residual Connections*.
*   **🏆 Scalable & Deep SNNs:** Breaks the depth limitations of prior normalization-free SNNs, delivering high-performance training across deep VGG, ResNet, and **Spiking Transformer** architectures.
*   **📊 State-of-the-Art Accuracy:** Reaches a competitive **68.05%** top-1 accuracy on **ImageNet** (T=4), matching or outperforming computationally heavy dynamic BN methods.
*   **🔋 Edge-Hardware Friendly:** Reduces FPGA lookup table (LUT) resource consumption for neuron implementations by **96.4%**, paving the way for ultra-low-power neuromorphic deployment.

## 📄 Abstract
The performance of deep spiking neural networks (SNNs) often relies on batch normalization (BN). However, the advanced dynamic BN variants used in state-of-the-art models introduce runtime multiplications, which weaken the hardware-efficiency motivation of SNNs. To address this tension, we identify catastrophic firing-rate decay as a primary cause of severe performance degradation in normalization-free SNNs. Guided by this insight, this work proposes the Intrinsically Stable SNN (IS-SNN) architecture, which removes activation-normalization layers by enforcing signal homeostasis through topology-aware weight standardization and modified residual connections. By folding the standardization operations into static weights offline, IS-SNN removes the runtime statistics tracking and multiplications introduced by activation normalization, restoring an accumulation-oriented inference datapath. Comprehensive experiments show that IS-SNN achieves performance competitive with or superior to computationally expensive dynamic BN techniques across VGG, ResNet, and Transformer-based models. Notably, it achieves a competitive accuracy of 68.05% on ImageNet and overcomes the severe depth limitations of prior BN-free attempts. Together with a 96.4% reduction in FPGA lookup table resource consumption for neuron implementations, these results support IS-SNN as a practical framework for building accurate and hardware-friendly deep neuromorphic systems.

## 🛠️ Requirements
- python==3.10
- numpy==1.23.5
- spikingjelly==0.0.0.0.14
- torch==2.2.0
- torchvision==0.17.0
- timm==1.0.7

### Environment Setup

We recommend using Anaconda to create a virtual environment:

```bash
conda create -n issnn python=3.10
conda activate issnn
```

Install PyTorch and dependencies:

```bash
# Install PyTorch (Choose based on your CUDA version)
# CUDA 11.8
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install SpikingJelly and timm
pip install timm==1.0.7
pip install spikingjelly==0.0.0.0.14
```

## 💻 Usage
Training on the CIFAR-10 dataset with AMP and Mixup/Cutmix: 
``` bash
CUDA_VISIBLE_DEVICES=0 \
python train.py --batch_size 128 --dataset_path '/ssd/Datasets/CIFAR10/' --dataset 'cifar10'\
                --class_number 10 --epochs 256 --lr 0.02 --weight_decay 5e-4 --amp\
                --timestep 4 --alpha 0.5 --mixup --workers 4 --print_freq 30 --name 'CIFAR10'
```
Training on the ImageNet dataset with AMP: 
``` bash
python train.py --batch_size 256 --dataset_path '/ssd/Datasets/ImageNet/' --dataset 'imagenet'\
                --class_number 1000 --epochs 128 --lr 0.2 --weight_decay 0 --amp\
                --timestep 4 --alpha 0.5 --workers 16 --print_freq 200 --name 'ImageNet'\
                --multiprocessing_distributed
```

## 📊 Main Results

Here is the performance summary of **IS-SNN** across various standard datasets and network architectures. By removing activation-normalization layers, IS-SNN achieves competitive accuracy with zero runtime normalization overhead.

<table>
  <thead>
    <tr>
      <th align="center">Dataset</th>
      <th align="center">Architecture</th>
      <th align="center">Timestep</th>
      <th align="center">Accuracy (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr align="center">
      <td rowspan="1" align="center" style="vertical-align: middle;"><b>ImageNet</b></td>
      <td>SEW-ResNet-34</td>
      <td>4</td>
      <td><b>68.05</b></td>
    </tr>
    <tr align="center">
      <td rowspan="2" align="center" style="vertical-align: middle;"><b>CIFAR-10</b></td>
      <td>VGG-11</td>
      <td>4</td>
      <td><b>95.06</b></td>
    </tr>
    <tr align="center">
      <td>SEW-ResNet-19</td>
      <td>6 / 4 / 2</td>
      <td><b>96.12 / 96.02 / 95.65</b></td>
    </tr>
    <tr align="center">
      <td rowspan="2" align="center" style="vertical-align: middle;"><b>CIFAR-100</b></td>
      <td>VGG-11</td>
      <td>4</td>
      <td><b>77.13</b></td>
    </tr>
    <tr align="center">
      <td>SEW-ResNet-19</td>
      <td>6 / 4 / 2</td>
      <td><b>80.72 / 79.97 / 79.03</b></td>
    </tr>
    <tr align="center">
      <td rowspan="1" align="center" style="vertical-align: middle;"><b>DVS-Gesture</b></td>
      <td>SEW-7B-Net</td>
      <td>16</td>
      <td><b>96.88</b></td>
    </tr>
  </tbody>
</table>
<br>

We provide pre-trained models below.
- [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/Ruichen0424/IS-SNN)


## 📜 Citation
If you find our code useful for your research, or use the IS-SNN architecture, please consider citing:

```bibtex
@article{ma2026intrinsically,
  title={Intrinsically Stable Spiking Neural Networks: Overcoming the Performance Barrier in the Absence of Batch Normalization},
  author={Ma, Ruichen and Zhang, Xiaoyang and Bai, Jian and Qiao, Guanchao and Meng, Liwei and Ning, Ning and Liu, Yang and Hu, Shaogang},
  journal={arXiv preprint arXiv:2606.31695},
  year={2026}
}
```