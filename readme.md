# Reproduce Tips for ASGL 

---

## Tips:

* The surrogate forwarding functions $H_\alpha(x)$ could be found in model/activation，including shift clipping, sigmoid, tanh et al.

* The implementation of hybrid forwarding with spike noise $\hat{H}_\alpha(\boldsymbol{x})=(1-\boldsymbol{m}) \odot H_\alpha(\boldsymbol{x})+\boldsymbol{m} \odot \Phi(\Theta(\boldsymbol{x}))$ could be found in the python class  model/activation/EfficientNoisySpikeII

* Try to reproduce the results on the CIFAR-100 dataset with such a command:

  ```python
  python main.py --seed 60 --arch resnet18  --auto_aug --cutout --wd 5e-4 --dataset CIFAR100 --act mns_sig  --T 2 --decay 0.5  --data_path /data2/wzm/cifar100   --bn_type tdbn  --alpha 5.0    --p 0.2   --gamma 1.0 
  ```

  * --act: specifies the surrogate forwarding function to use

  *  --p: the noise probability which denotes the ratio of analog mode

  * --gamma: the decay rate of $p$ during training

  * --ns_milestone: the milestones to adjust $p$

    



 

 

## 