import torch
import copy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.activation import EfficientNoisySpike


class MutiStepNoisyRateScheduler:

    def __init__(self, init_p=1, reduce_ratio=0.9, milestones=[0.3, 0.7, 0.9, 0.95], num_epoch=100, start_epoch=0):
        self.reduce_ratio = reduce_ratio
        self.p = init_p
        self.milestones = [int(m * num_epoch) for m in milestones]
        self.num_epoch = num_epoch
        self.start_epoch = start_epoch

    def set_noisy_rate(self, p, model):
        for m in model.modules():
            if isinstance(m, EfficientNoisySpike):
                m.p = p

    def __call__(self, epoch, model):
        for one in self.milestones:
            if one + self.start_epoch == epoch:
                self.p *= self.reduce_ratio
                print('change noise rate as ' + str(self.p))
                self.set_noisy_rate(self.p, model)
                break
