# Adaptive Smoothing Gradient Learning for Spiking Neural Networks

---

This repository contains a Pytorch Implementation of **Adaptive Smoothing Gradient Learning for Spiking Neural Networks, ICML 2023** 
> See more from the paper [[PMLR\]](https://proceedings.mlr.press/v202/wang23j.html) or [[OpenReview\]](https://openreview.net/forum?id=GdkwSGTpbC).

**Table of contents:**

- [Abstract](#abstract)
- [Dependency](#dependency)
- [Directory Tree](#directory)
- [Usage](#usage)
- [Citation](#citation)

## Abstract

To train SNNs directly considering the all-or-none nature of spikes,  existing solutions introduce additional smoothing error on spike firing which leads to the gradients being estimated inaccurately. This work explores how to adaptively adjust the relaxation degree and eliminate smoothing error progressively. Here, we propose a methodology such that training a prototype neural network will evolve into training an SNN gradually by fusing the learnable relaxation degree into the network with random spike noise. In this way, the network learns adaptively the accurate gradients of loss landscape in SNNs.

<img src="doc/figure/arch_v6.png" alt="introduction_figure" style="zoom:100%;" />

| Width ($\alpha$) | CIFAR-10 | CIFAR-10  |      | CIFAR-100 | CIFAR-100 |
| :--------------: | :------: | :-------: | :--: | :-------: | :-------: |
|                  |    SG    | **ASGL**  |      |    SG     | **ASGL**  |
|       0.5        |  93.19   | **94.11** |      |   75.76   | **76.54** |
|       1.0        |  93.78   | **94.30** |      |   65.19   | **76.09** |
|       2.5        |  90.68   | **94.09** |      |   15.12   | **76.18** |
|       5.0        |  62.34   | **93.61** |      |   8.04    | **76.68** |
|       10.0       |  30.85   | **93.53** |      |   6.14    | **76.00** |


## Dependency

The major dependencies of this repo are listed as below. 

```
# Name                 Version
python                  3.8.12 
numpy                   1.21.2
scipy                   1.7.3
scikit-learn            1.0.2
cudatoolkit             11.6
torch                   1.11.0
torchaudio              0.11.0
torchvision             0.12.0
tensorboard             2.6.0
```
## Directory Tree

```
.
├── experiment
│   ├── development
│   │   ├── config
│   │   ├── log
│   │   └── main.py
│   └── dvs
│       ├── config
│       ├── log
│       └── main.py
├── model
│   ├── activation.py
│   ├── cell.py
│   ├── layer.py
│   ├── loss.py
│   ├── resnet.py
│   ├── scheduler.py
│   └── vgg.py
└── util
    ├── data.py
    ├── image_augment.py
    └── util.py

```
The experiment code and model definition for static image, asynchronous event stream and sounds are located on corresponding directories in experiment (CIFAR-10/CIFAR100 $\to$ `experiment/evelopment`, Gesture/DVS-CIFAR10 $\to$ `experiment/dvs`,  ). The corresponding checkpoint and log file could be found in ./checkpoint and ./log, respectively. The proposed *Adaptive Smoothing Gradient Learning (ASGL)* implementation can be found in the python class  `model/activation/EfficientNoisySpikeII`. All the surrogate functions for both forward calculation $H_\alpha(x)$ (used in *ASGL*) and backward propagation $h_\alpha(x)$ (used in *SG*) can be found in `model/activation`.

## Usage

1. Try to reproduce the results on the CIFAR-100 dataset with the following command:
    ```bash
    python development/main.py --seed 60 --arch resnet18  --auto_aug --cutout --wd 5e-4 --dataset CIFAR100 --act mns_sig  --T 2 --decay 0.5 --thresh 1.0 --data_path [your datapath]   --bn_type tdbn  --alpha 5.0    --p 0.2   --gamma 1.0
    ```
    ASGL using the above hyperparameters achieve a performance of 76.76% on the CIFAR-100 dataset 	under 2 time steps (higher than the reported performance in the manuscript) even for an aggressive width setting $\alpha=5$. 
    
2. Try to reproduce the results on the DVS-CIFAR10 dataset with the following command:
    ```bash
    python dvs/main.py --seed 200 --arch VGGSNN2  --bn_type tdbn --wd 1e-3 --num_workers 8  --act mns_rec --decay 0.5  --p 0.1 --gamma 1  --alpha 1.0   --train_width --dataset 
    ```
    
    ASGL using the above hyperparameters achieve a performance of 85.50% on the DVS-CIFAR10 dataset under 10 time steps (higher than the reported performance in the manuscript). 
    
2. Some explanation for hyper-parameters

   * --p: the noise probability which denotes the ratio of analog mode

   * --gamma: the decay rate of $p$ during training

   * --ns_milestone: the milestones to adjust $p$

3. The surrogate forwarding functions $H_\alpha(x)$ could be found in `model/activation`

4. The implementation of hybrid forwarding with spike noise $\hat{H}_\alpha(\mathbf{x})$ could be found in the python class  `model/activation/EfficientNoisySpikeII`

## Citation: 

Please consider citing the following article if you find this work useful for your research.

```tex
@inproceedings{wang2023adaptive,
  title={Adaptive Smoothing Gradient Learning for Spiking Neural Networks},
  author={Wang, Ziming and Jiang, Runhao and Lian, Shuang and Yan, Rui and Tang, Huajin},
  booktitle={International Conference on Machine Learning},
  organization={PMLR}
}
```

## Updating ...

