# Adaptive Smoothing Gradient Learning for Spiking Neural Networks

---

This repository contains a Pytorch Implementation of **Adaptive Smoothing Gradient Learning for Spiking Neural Networks, ICML 2023**.

## Tips:

1. The surrogate forwarding functions $H_\alpha(x)$ could be found in model/activation

2. The implementation of hybrid forwarding with spike noise $\hat{H}_\alpha(\mathbf{x})=(1-\mathbf{m}) \odot H_\alpha(\mathbf{x})+\mathbf{m} \odot \Phi(\Theta(\mathbf{x}))$ could be found in the python class  model/activation/EfficientNoisySpikeII

3. Try to reproduce the results on the CIFAR-100 dataset with the following command:

```python
python main.py --seed 60 --arch resnet18  --auto_aug --cutout --wd 5e-4 --dataset CIFAR100 --act mns_sig  --T 2 --decay 0.5 --thresh 1.0 --data_path [your datapath]   --bn_type tdbn  --alpha 5.0    --p 0.2   --gamma 1.0
```
ASGL using the above hyperparameters achieve a performance of 76.76% on the CIFAR-100 dataset under 2 time steps (higher than the reported performance in the manuscript) even for an aggressive width setting $\alpha=5$. The corresponding checkpoint and log file could be found in ./checkpoint and ./log, respectively.

4. Some explanation for hyper-parameters
      * --p: the noise probability which denotes the ratio of analog mode
      * --gamma: the decay rate of $p$ during training
      * --ns_milestone: the milestones to adjust $p$

## Citation info: 

Please cite this paper using the following BibTeX entry if you find this work useful for your research.

```tex
@inproceedings{wang2023adaptive,
  title={Adaptive Smoothing Gradient Learning for Spiking Neural Networks},
  author={Wang, Ziming and Jiang, Runhao and Lian, Shuang and Yan, Rui and Tang, Huajin},
  booktitle={International Conference on Machine Learning},
  organization={PMLR}
}
```

## Updating ...

