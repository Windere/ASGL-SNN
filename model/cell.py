# -*- coding: utf-8 -*-

import abc
import copy
import torch
import torch.nn as nn
import torch.jit as jit
from model.activation import  EfficientNoisySpikeII
import torch.nn.functional as F
import copy


class LIFCell(nn.Module):
    """
        simulating iterative leaky-integrate-and-fire neurons and mapping input currents into output spikes
    """

    def __init__(self, spike_fn, decay=None, thresh=None, vreset=None, use_gate=False):
        super(LIFCell, self).__init__()
        self.decay = copy.deepcopy(decay)
        self.thresh = copy.deepcopy(thresh)
        self.vreset = copy.deepcopy(vreset)
        self._reset_parameters()
        self.use_gate = use_gate
        self.spike_fn = copy.deepcopy(spike_fn)

    def forward(self, vmem, psp):
        # print(self.thresh)
        # print(F.sigmoid(self.decay))
        # if isinstance(self.spike_fn, EfficientNoisySpike):
        #     psp /= self.spike_fn.inv_sg.alpha
        # print('teste')
        gates = None
        # if self.use_gate:
        #     psp, gates = torch.chunk(psp, 2, dim=1)
            # gates = torch.sigmoid(gates)
        vmem = torch.sigmoid(self.decay) * vmem + psp
        if isinstance(self.spike_fn, EfficientNoisySpikeII):  # todo: check here
            # print('trigger!')
            self.spike_fn.reset_mask()
        if self.use_gate:
            spike = self.spike_fn(vmem - self.thresh, gates)
        else:
            spike = self.spike_fn(vmem - self.thresh)
        if self.vreset is None:
            vmem -= self.thresh * spike
        else:
            vmem = vmem * (1 - spike) + self.vreset * spike
        # spike *= self.thresh
        return vmem, spike

    def _reset_parameters(self):
        if self.thresh is None:
            self.thresh = 0.5
        if self.decay is None:
            self.decay = nn.Parameter(torch.Tensor([0.9]))

    def reset(self):
        pass
        # if isinstance(self.decay, nn.Parameter):
        #     self.decay.data.clamp_(0., 1.)
        if isinstance(self.thresh, nn.Parameter):
            self.thresh.data.clamp_(min=0.)
