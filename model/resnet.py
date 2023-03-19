# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from model.layer import *


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def warpBN(channel, bn_type, nb_steps):
    if bn_type == 'tdbn':
        return tdLayer(nn.BatchNorm2d(channel), nb_steps)
    elif bn_type == 'bntt':
        return TemporalBN(channel, nb_steps, step_wise=True)
    elif bn_type == '':
        return TemporalBN(channel, nb_steps, step_wise=False)
    elif bn_type == 'idnt':
        return nn.Identity()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn_type='', expand=1, **kwargs_spikes):
        super(BasicBlock, self).__init__()
        self.kwargs_spikes = kwargs_spikes
        self.nb_steps = kwargs_spikes['nb_steps']
        self.expand = expand
        self.conv1 = tdLayer(nn.Conv2d(in_planes, planes * expand, kernel_size=3, stride=stride, padding=1, bias=False),
                             self.nb_steps)
        self.bn1 = warpBN(planes * expand, bn_type, self.nb_steps)
        self.spike1 = LIFLayer(**kwargs_spikes)
        self.conv2 = tdLayer(nn.Conv2d(planes, planes * expand, kernel_size=3, stride=1, padding=1, bias=False),
                             self.nb_steps)
        self.bn2 = warpBN(planes * expand, bn_type, self.nb_steps)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                tdLayer(
                    nn.Conv2d(in_planes, planes * self.expansion * expand, kernel_size=1, stride=stride, bias=False),
                    self.nb_steps),
                warpBN(self.expansion * planes * expand, bn_type, self.nb_steps)
                # tdBatchNorm(nn.BatchNorm2d(planes * BasicBlock.expansion), alpha=1 / math.sqrt(2.))
            )
        self.spike2 = LIFLayer(**kwargs_spikes)

    def forward(self, x):
        out = self.spike1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # if self.expand != 1 and len(self.shortcut) == 0:
        #     x = torch.cat([x, x], dim=2)
        #     # todo: check here
        out += self.shortcut(x)
        out = self.spike2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bn_type='', **kwargs_spikes):
        super(Bottleneck, self).__init__()
        self.kwargs_spikes = kwargs_spikes
        self.nb_steps = kwargs_spikes['nb_steps']
        self.conv1 = tdLayer(nn.Conv2d(in_planes, planes, kernel_size=1, bias=False), self.nb_steps)
        self.bn1 = warpBN(planes, bn_type, self.nb_steps)
        self.spike1 = LIFLayer(**kwargs_spikes)
        self.conv2 = tdLayer(nn.Conv2d(planes, planes, kernel_size=3,
                                       stride=stride, padding=1, bias=False), self.nb_steps)
        self.bn2 = warpBN(planes, bn_type, self.nb_steps)
        self.spike2 = LIFLayer(**kwargs_spikes)
        self.conv3 = tdLayer(nn.Conv2d(planes, self.expansion *
                                       planes, kernel_size=1, bias=False), self.nb_steps)
        self.bn3 = warpBN(self.expansion *
                          planes, bn_type, self.nb_steps)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                tdLayer(nn.Conv2d(in_planes, self.expansion * planes,
                                  kernel_size=1, stride=stride, bias=False), self.nb_steps),
                warpBN(self.expansion * planes, bn_type, self.nb_steps)
            )
        self.spike3 = LIFLayer(**kwargs_spikes)

    def forward(self, x):
        out = self.spike1(self.bn1(self.conv1(x)))
        out = self.spike2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.spike3(out)
        return out


class ResNet19(nn.Module):
    def __init__(self, block, num_block_layers, num_classes=10, in_channel=3, bn_type='', **kwargs_spikes):
        super(ResNet19, self).__init__()
        self.in_planes = 128
        self.bn_type = bn_type
        self.kwargs_spikes = kwargs_spikes
        self.nb_steps = kwargs_spikes['nb_steps']
        self.conv0 = nn.Sequential(
            tdLayer(nn.Conv2d(in_channel, self.in_planes, kernel_size=3, padding=1, stride=1, bias=False),
                    nb_steps=self.nb_steps),
            warpBN(self.in_planes, bn_type, self.nb_steps),
            LIFLayer(**kwargs_spikes)
        )
        self.layer1 = self._make_layer(block, 128, num_block_layers[0], stride=1)
        self.layer2 = self._make_layer(block, 256, num_block_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, num_block_layers[2], stride=2)
        self.avg_pool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)), nb_steps=self.nb_steps)
        self.classifier = nn.Sequential(
            tdLayer(nn.Linear(512 * block.expansion, 256, bias=False), nb_steps=self.nb_steps),
            LIFLayer(**kwargs_spikes),
            tdLayer(nn.Linear(256, num_classes, bias=False), nb_steps=self.nb_steps),
            Readout()
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.bn_type, **self.kwargs_spikes))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out, _ = torch.broadcast_tensors(x, torch.zeros((self.nb_steps,) + x.shape))
        out = self.conv0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.shape[0], out.shape[1], -1)
        out = self.classifier(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_block_layers, num_classes=10, in_channel=3, bn_type='', **kwargs_spikes):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.bn_type = bn_type
        self.kwargs_spikes = kwargs_spikes
        self.nb_steps = kwargs_spikes['nb_steps']
        self.conv0 = nn.Sequential(
            tdLayer(nn.Conv2d(in_channel, self.in_planes, kernel_size=3, padding=1, stride=1, bias=False),
                    nb_steps=self.nb_steps),
            warpBN(self.in_planes, bn_type, self.nb_steps),
            LIFLayer(**kwargs_spikes)
        )
        self.layer1 = self._make_layer(block, 64, num_block_layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_block_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_block_layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_block_layers[3], stride=2)

        self.avg_pool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)), nb_steps=self.nb_steps)
        self.classifier = nn.Sequential(
            tdLayer(nn.Linear(512 * block.expansion, num_classes), nb_steps=self.nb_steps),
            Readout()
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.bn_type, **self.kwargs_spikes))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out, _ = torch.broadcast_tensors(x, torch.zeros((self.nb_steps,) + x.shape))
        out = self.conv0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.shape[0], out.shape[1], -1)
        out = self.classifier(out)
        return out


depth_lst = [18, 34, 50, 101, 152]  # use the standard resnet series for imagenet or modified structure from tdbn


def cfg(depth):
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 19, 34, 50, 101, 152"
    cf_dict = {
        18: (BasicBlock, [2, 2, 2, 2]),
        34: (BasicBlock, [3, 4, 6, 3]),
        50: (Bottleneck, [3, 4, 6, 3]),
        101: (Bottleneck, [3, 4, 23, 3]),
        152: (Bottleneck, [3, 8, 36, 3]),
    }
    return cf_dict[depth]


def ResNetX(depth, num_class=10, in_channel=3, bn_type='', use_gates=False, **kwargs_spikes):
    if depth in depth_lst:
        block, num_blocks = cfg(depth)
        if use_gates:
            raise NotImplementedError('this part will be released latter')
        else:
            return ResNet(block, num_blocks, num_class, in_channel=in_channel, bn_type=bn_type, **kwargs_spikes)
        # if modified:
        #     return ResNet_Modified(block, num_blocks, num_class, act_fun=act_fun)
        # else:
        #     return ResNet(block, num_blocks, num_class, act_fun=act_fun)

    elif depth == 19:
        return ResNet19(BasicBlock, [3, 3, 2], num_class, in_channel=in_channel, bn_type=bn_type, **kwargs_spikes)
