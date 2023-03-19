# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import numpy as np


class InvSigmoid(nn.Module):
    def __init__(self, alpha: float = 1.0, learnable=True):
        super(InvSigmoid, self).__init__()
        self.learnable = learnable
        self.alpha = alpha

    def get_temperature(self):
        return self.alpha.detach().clone()

    def forward(self, x, gates=None):
        if self.learnable and not isinstance(self.alpha, nn.Parameter):
            self.alpha = nn.Parameter(torch.Tensor([self.alpha]).to(x.device))
        if gates is None:
            return torch.sigmoid(self.alpha * x)
        else:
            raise NotImplementedError('gates is not supported now')


class EfficientNoisySpike(nn.Module):
    def __init__(self, inv_sg=InvSigmoid()):
        super(EfficientNoisySpike, self).__init__()
        self.inv_sg = inv_sg

    def forward(self, x, gates=None):
        return self.inv_sg(x, gates) + ((x >= 0).float() - self.inv_sg(x, gates)).detach()


class EfficientNoisySpikeII(EfficientNoisySpike):  # todo: write ABC
    def __init__(self, inv_sg=InvSigmoid(), p=0.5, spike=True):
        super(EfficientNoisySpikeII, self).__init__()
        self.inv_sg = inv_sg
        self.p = p
        self.spike = spike
        self.reset_mask()

    def create_mask(self, x: torch.Tensor):
        return torch.bernoulli(torch.ones_like(x) * (1 - self.p))

    def forward(self, x, gates=None):
        sigx = self.inv_sg(x, gates)
        if self.training:
            if self.mask is None:
                self.mask = self.create_mask(x)
            return sigx + (((x >= 0).float() - sigx) * self.mask).detach()
            # return sigx * (1 - self.mask) + ((x >= 0).float() * self.mask).detach()
        if self.spike:
            return (x >= 0).float()
        else:
            return sigx

    def reset_mask(self):
        self.mask = None
