# -*- coding: utf-8 -*-
"""
@File: vgg.py

@Author: Ziming Wang

@Time: 2022/5/24 19:37

@Usage:  
"""
import math
import torch.nn as nn
from model.layer import *

feature_cfg = {
    'VGG5': [64, 'A', 128, 128, 'A', 'AA'],
    'VGG9': [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 'A', 512, 'A', 512, 512, 'AA'],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512, 'AA'],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512],
    'CIFAR': [128, 256, 'A', 512, 'A', 1024, 512],
    'VGGSNN2': [64, 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512, 'A'],
}

clasifier_cfg = {
    'VGG16': [2048, 4096, 4096, 10],
    'VGG5': [128, 10],
    'VGG11': [512, 10],
    'VGG13': [512, 10],
    'VGG19': [2048, 4096, 4096, 10],
    'VGGSNN2': [4608, 10]
}


# todo: provide a promising snn dropout
class VGG(nn.Module):
    def __init__(self, architecture='VGG16', kernel_size=3, in_channel=3, use_bias=True,
                 bn_type=None, num_class=10, readout_mode='psp_avg',
                 **kwargs_spikes):
        super(VGG, self).__init__()
        self.kwargs_spikes = kwargs_spikes
        self.nb_steps = kwargs_spikes['nb_steps']
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.use_bias = use_bias
        self.bn_type = bn_type
        self.readout_mode = readout_mode
        self.num_class = num_class
        clasifier_cfg[architecture][-1] = num_class
        self.feature = self._make_feature(feature_cfg[architecture])
        self.classifier = self._make_classifier(clasifier_cfg[architecture])
        self._initialize_weights()

    def _make_feature(self, config):
        layers = []
        channel = self.in_channel
        for x in config:
            if x == 'A':
                layers.append(tdLayer(nn.AvgPool2d(kernel_size=2, stride=2), self.nb_steps))
            elif x == 'AA':
                layers.append(tdLayer(nn.AdaptiveAvgPool2d((1, 1)), self.nb_steps))

            else:
                layers.append(tdLayer(nn.Conv2d(in_channels=channel, out_channels=x, kernel_size=self.kernel_size,
                                                stride=1, padding=self.kernel_size // 2, bias=self.use_bias),
                                      nb_steps=self.nb_steps))
                if self.bn_type == 'tdbn':
                    layers.append(tdLayer(nn.BatchNorm2d(x), self.nb_steps))
                elif self.bn_type == 'bntt':
                    layers.append(TemporalBN(x, self.nb_steps, step_wise=True))
                elif self.bn_type == 'bn':
                    layers.append(TemporalBN(x, self.nb_steps, step_wise=False))
                layers.append(LIFLayer(**self.kwargs_spikes))
                channel = x
        return nn.Sequential(*layers)

    def _make_classifier(self, config):
        layers = []
        for i in range(len(config) - 1):
            layers.append(tdLayer(nn.Linear(config[i], config[i + 1], bias=self.use_bias), nb_steps=self.nb_steps))
            layers.append(LIFLayer(**self.kwargs_spikes))
        layers.pop()
        # layers.append(Readout(self.readout_mode)) # comment this line for TET loss
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0, 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.5)
                # m.weight.data.normal_(0, 0.5)
                # n = m.weight.size(1)
                # m.weight.data.normal_(0, 1.0 / float(n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if x.dim() <= 4:
            out, _ = torch.broadcast_tensors(x, torch.zeros((self.nb_steps,) + x.shape))
        else:
            out = x.permute(1, 0, 2, 3, 4)
        out = self.feature(out)
        out = out.view(out.shape[0], out.shape[1], -1)
        out = self.classifier(out)
        return out


class VGG_block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias, **kwargs_spikes):
        super(VGG_block, self).__init__()
        self.conv = tdLayer(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                      stride=stride, bias=bias), nb_steps=kwargs_spikes['nb_steps'])
        # self.bn = tdBatchNorm(nn.BatchNorm2d(out_channel), 1)
        self.bn = TemporalBN(in_channels=out_channel, nb_steps=kwargs_spikes['nb_steps'])
        self.spike = LIFLayer(**kwargs_spikes)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.spike(out)

        return out


class bVGG(nn.Module):
    def __init__(self, vgg_name='VGG16', labels=10, dataset='CIFAR10', kernel_size=3, dropout=0,
                 use_bias=True, **kwargs_spikes):
        super(bVGG, self).__init__()
        self.kwargs_spikes = kwargs_spikes
        self.nb_steps = kwargs_spikes['nb_steps']
        self.dataset = dataset
        self.kernel_size = kernel_size
        self.dropout = dropout  # todo: use for snn
        self.use_bias = use_bias
        self.vgg_name = vgg_name
        self.features = self._make_layers(feature_cfg[vgg_name])
        if vgg_name != 'VGG5' and dataset != 'MNIST':
            self.classifier = nn.Sequential(
                tdLayer(nn.Linear(512 * 2 * 2, 4096, bias=use_bias), nb_steps=self.nb_steps),
                LIFLayer(**kwargs_spikes),
                tdLayer(nn.Linear(4096, 4096, bias=use_bias), nb_steps=self.nb_steps),
                LIFLayer(**kwargs_spikes),
                # nn.Dropout(0.5),
                tdLayer(nn.Linear(4096, labels, bias=use_bias), nb_steps=self.nb_steps),
                Readout(),
                # nn.Dropout(0.5),
                # tdLayer(nn.Linear(512, labels, bias=True)),
                # LIF()
            )
        self._initialize_weights2()

    def forward(self, x):
        x = self.features[1](self.features[0](x))  # todo: 真的有用吗？
        # print(x.size())
        out, _ = torch.broadcast_tensors(x, torch.zeros((self.nb_steps,) + x.shape))
        # print(out.size())
        # out = out.permute(1, 2, 3, 4, 0)
        # print(out.size())
        # out = self.features[3:](out)
        out = self.features[2:](out)
        # print(out.shape)
        out = out.view(out.shape[0], out.shape[1], -1)
        # print(out.shape)
        out = self.classifier(out)
        return out

    def _initialize_weights2(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # m.weight.data.normal_(0, 0.5)
                # n = m.weight.size(1)
                # m.weight.data.normal_(0, 1.0 / float(n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layers(self, cfg):
        layers = []
        k = 0
        if self.dataset == 'MNIST':
            in_channels = 1
        else:
            in_channels = 3
        for x in cfg:
            stride = 1

            if x == 'A':
                # layers.pop()
                layers.append(tdLayer(nn.AvgPool2d(kernel_size=2, stride=2), nb_steps=self.nb_steps))
            else:
                if k == 0:
                    layers.append(
                        nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2,
                                  stride=stride, bias=self.use_bias))
                    layers.append(nn.BatchNorm2d(x))
                    # layers.append(LIF(**self.kwargs_spikes))
                # layers += [tdLayer(nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2,
                #                      stride=stride, bias=False)),
                #            tdBatchNorm(nn.BatchNorm2d(x), 1),
                #            LIF()
                #            ]
                else:
                    layers.append(
                        VGG_block(in_channels, x, kernel_size=self.kernel_size, stride=stride, bias=self.use_bias,
                                  **self.kwargs_spikes))
                in_channels = x
                k += 1
        if self.vgg_name == 'CIFAR':
            layers.append(tdLayer(nn.AdaptiveAvgPool2d((2, 2)), nb_steps=self.nb_steps)),

        return nn.Sequential(*layers)


if __name__ == '__main__':
    kwargs_spikes = {'nb_steps': 2}
    model = VGG(**kwargs_spikes)
    print(model)
