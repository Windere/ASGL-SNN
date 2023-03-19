# -*- coding: utf-8 -*-

import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import csv
import time
import torch
from model.resnet import ResNetX
import os
import re
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from util.util import setup_seed, Logger
from torch.utils.tensorboard import SummaryWriter
from util.image_augment import CIFAR10Policy, Cutout
from torchvision import datasets, transforms, models
from model.activation import EfficientNoisySpike, EfficientNoisySpikeII, InvSigmoid
from model.cell import LIFCell
from model.scheduler import MutiStepNoisyRateScheduler


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

def runTrain(epoch, train_ldr, optimizer, model, evaluator, args=None, encoder=None):
    loss_record = []
    predict_tot = []
    label_tot = []
    model.train()
    start_time = time.time()
    for idx, (ptns, labels) in enumerate(train_ldr):
        ptns, labels = ptns.to(args.device), labels.to(args.device)
        if encoder is not None:
            ptns = encoder(ptns)
        # optimizer.zero_grad()
        if isinstance(optimizer, list):
            for optim in optimizer:
                optim.zero_grad()
        output = model(ptns)
        loss = evaluator(output, labels)
        loss.backward()
        # optimizer.step()
        if isinstance(optimizer, list):
            for optim in optimizer:
                optim.step()
        else:
            optimizer.step()
        predict = torch.argmax(output, axis=1)
        # record results
        loss_record.append(loss.detach().cpu())
        predict_tot.append(predict)
        label_tot.append(labels)
        if (idx + 1) % args.log_interval == 0:
            log.info('\nEpoch [%d/%d], Step [%d/%d], Loss: %.5f'
                     % (
                         epoch, args.num_epoch + args.start_epoch, idx + 1,
                         len(train_ldr.dataset) // args.train_batch_size,
                         loss_record[-1] / args.train_batch_size))
            log.info('Time elasped: {}'.format(time.time() - start_time))
    predict_tot = torch.cat(predict_tot)
    label_tot = torch.cat(label_tot)
    train_acc = torch.mean((predict_tot == label_tot).float())
    train_loss = torch.tensor(loss_record).sum() / len(label_tot)
    return train_acc, train_loss


def runTest(val_ldr, model, evaluator, args=None, encoder=None):
    model.eval()
    with torch.no_grad():
        predict_tot = {}
        label_tot = []
        loss_record = []
        key = 'ann' if encoder is None else 'snn'
        for idx, (ptns, labels) in enumerate(val_ldr):
            # ptns: batch_size x num_channels x T x nNeu ==> batch_size x T x (nNeu*num_channels)
            ptns, labels = ptns.to(args.device), labels.to(args.device)
            if encoder is not None:
                ptns = encoder(ptns)
            output = model(ptns)
            if isinstance(output, dict):
                for t in output.keys():
                    if t not in predict_tot.keys():
                        predict_tot[t] = []
                    predict = torch.argmax(output[t], axis=1)
                    predict_tot[t].append(predict)
                loss = evaluator(output[encoder.nb_steps], labels)

            else:
                if key not in predict_tot.keys():
                    predict_tot[key] = []
                loss = evaluator(output, labels)
                # snn.clamp()
                predict = torch.argmax(output, axis=1)
                predict_tot[key].append(predict)
            loss_record.append(loss)
            label_tot.append(labels)

        label_tot = torch.cat(label_tot)
        val_loss = torch.tensor(loss_record).sum() / len(label_tot)
        if 'ann' not in predict_tot.keys() and 'snn' not in predict_tot.keys():
            val_acc = {}
            for t in predict_tot.keys():
                val_acc[t] = torch.mean((torch.cat(predict_tot[t]) == label_tot).float())

        else:
            predict_tot = torch.cat(predict_tot[key])
            val_acc = torch.mean((predict_tot == label_tot).float())
        return val_acc, val_loss


def loadData(name, root, cutout=False, auto_aug=False):
    num_class, normalize, train_data, test_data = None, None, None, None
    train_transform = []
    if name == 'CIFAR10' or name == 'CIFAR100':
        train_transform = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if auto_aug:
        train_transform.append(CIFAR10Policy())
    train_transform.append(transforms.ToTensor())
    if cutout:
        train_transform.append(Cutout(n_holes=1, length=16))
    if name == 'CIFAR10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        num_class = 10
    elif name == 'CIFAR100':
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        num_class = 100
    elif name == 'MNIST':
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        num_class = 10
    train_transform.append(normalize)
    train_transform = transforms.Compose(train_transform)
    val_transform = transforms.Compose([transforms.ToTensor(),
                                        normalize
                                        ])
    if name == 'CIFAR100':
        train_data = datasets.CIFAR100(root=root, train=True, download=True,
                                       transform=train_transform)
        val_data = datasets.CIFAR100(root=root, train=False, download=True,
                                     transform=val_transform)
    elif name == 'CIFAR10':
        train_data = datasets.CIFAR10(root=root, train=True, download=True,
                                      transform=train_transform)
        val_data = datasets.CIFAR10(root=root, train=False, download=True,
                                    transform=val_transform)
    elif name == 'MNIST':
        train_data = datasets.MNIST(root=root, train=True, download=True,
                                    transform=train_transform)
        val_data = datasets.MNIST(root=root, train=False, download=True,
                                  transform=val_transform)
    return train_data, val_data, num_class


def warp_decay(decay):
    import math
    return torch.tensor(math.log(decay / (1 - decay)))


def split_params(model, paras=([], [], [])):
    for n, module in model._modules.items():
        if isinstance(module, LIFCell) and hasattr(module, "thresh"):
            for name, para in module.named_parameters():
                paras[0].append(para)
        elif 'batchnorm' in module.__class__.__name__.lower():
            for name, para in module.named_parameters():
                paras[2].append(para)
        elif isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.modules.conv._ConvNd):
            paras[1].append(module.weight)
            if module.bias is not None:
                paras[2].append(module.bias)
        elif len(list(module.children())) > 0:
            paras = split_params(module, paras)
        elif module.parameters() is not None:
            for name, para in module.named_parameters():
                paras[1].append(para)
    return paras


def get_temperatures(net):
    temperatures = []
    for m in net.modules():
        if isinstance(m, EfficientNoisySpike):
            temperatures.append(m.inv_sg.get_temperature())
    temperatures = torch.cat(temperatures).cpu()
    return temperatures


def main():
    start_epoch = 0
    best_epoch = 0
    best_acc = 0
    best_train_acc = 0
    train_trace, val_trace = dict(), dict()
    train_trace['acc'], train_trace['loss'], train_trace['temp'] = [], [], []
    val_trace['acc'], val_trace['loss'] = [], []
    writer = SummaryWriter(args.log_path)
    train_data, val_data, num_class = loadData(args.dataset, args.data_path, cutout=args.cutout, auto_aug=args.auto_aug)
    train_ldr = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.train_batch_size, shuffle=True,
                                            pin_memory=True, num_workers=args.num_workers)
    val_ldr = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                                          pin_memory=True,
                                          num_workers=args.num_workers)
    decay = nn.Parameter(warp_decay(args.decay))
    thresh = nn.Parameter(torch.tensor(args.thresh)) if args.train_thresh else args.thresh
    args.alpha = 1 / args.alpha
    if args.act == 'mns_sig':
        kwargs_spikes = {'nb_steps': args.T, 'vreset': 0, 'thresh': thresh,
                         'spike_fn': EfficientNoisySpikeII(p=args.p, inv_sg=InvSigmoid(alpha=args.alpha),
                                                           spike=True),
                         'decay': decay}
    depth = int(re.findall("\d+", args.arch)[0])
    model = ResNetX(depth, num_class, bn_type=args.bn_type, **kwargs_spikes).to(device, dtype)
    model(train_data[0][0].unsqueeze(0).to(device, dtype))
    log.info(model)
    params = split_params(model)
    spiking_params = [{'params': params[0], 'weight_decay': 0}]
    params = [
        {'params': params[1], 'weight_decay': args.wd},
        {'params': params[2], 'weight_decay': 0}
    ]
    if args.optim.lower() == 'sgdm':
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)
    elif args.optim.lower() == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, amsgrad=False)
    width_optim = optim.Adam(spiking_params, lr=3e-4)
    evaluator = torch.nn.CrossEntropyLoss()
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state['best_net'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['best_epoch']
        best_acc = state['best_acc']
        train_trace = state['traces']['train']
        val_trace = state['traces']['val']
        log.info('Load checkpoint from epoch {}'.format(start_epoch))
        log.info('Best accuracy so far {}.'.format(best_acc))
        log.info('Test the checkpoint: {}'.format(runTest(val_ldr, model, evaluator, args=args)))

    args.start_epoch = start_epoch
    if 'noisy_spike' in args.act or 'ns' in args.act:
        train_trace['temp'].append(get_temperatures(model))
    if args.scheduler.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.num_epoch)
    elif args.scheduler.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestone, gamma=0.8)
    rate_scheduler = MutiStepNoisyRateScheduler(init_p=args.p, reduce_ratio=args.gamma, milestones=args.ns_milestone,
                                                num_epoch=args.num_epoch, start_epoch=start_epoch)
    for epoch in tqdm(range(start_epoch, start_epoch + args.num_epoch)):
        train_acc, train_loss = runTrain(epoch, train_ldr, [optimizer, width_optim], model, evaluator, args=args)
        if args.scheduler != 'None':
            scheduler.step()
        val_acc, val_loss = runTest(val_ldr, model, evaluator, args=args)
        if args.p is not None:
            rate_scheduler(epoch, model)
        # saving checkpoint
        log.info('Epoch %d: train loss %.5f, train acc %.5f, test acc %.5f ' % (epoch, train_loss, train_acc, val_acc))
        # log.info('Evaluating custom_model: epoch {},  loss {}, acc {}'.format(epoch, val_loss, val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            best_train_acc = train_acc
            best_epoch = epoch
            log.info('Saving custom_model..  with acc {0} in the epoch {1}'.format(best_acc, epoch))
            state = {
                'best_acc': best_acc,
                'best_epoch': epoch,
                'best_net': model.state_dict(),
                'best_train_acc': best_train_acc,
                'optimizer': optimizer.state_dict(),
                'traces': {'train': train_trace, 'val': val_trace},
            }
            torch.save(state, os.path.join(args.ckpt_path, model_name + '.pth'))
        # record and log
        train_trace['acc'].append(train_acc)
        train_trace['loss'].append(train_loss)
        val_trace['acc'].append(val_acc)
        val_trace['loss'].append(val_loss)
        if 'noisy_spike' in args.act or 'ns' in args.act:
            temperature = get_temperatures(model)
            train_trace['temp'].append(temperature)
            print('Temperature at epoch {} : {}'.format(epoch, temperature.mean().item()))

        # record in tensorboard
        writer.add_scalars('Loss', {'val': val_loss, 'train': train_loss},
                           epoch + 1)
        writer.add_scalars('Acc', {'val': val_acc, 'train': train_acc},
                           epoch + 1)

    log.info('Finish training: the best validation accuracy of SNN is {} in epoch {}. \n The relate checkpoint path: {}'.format(
        best_acc, best_epoch, os.path.join(args.ckpt_path, model_name + '.pth')))


def get_model_name(model_name, args):
    aug_str = '_'.join(['cut' if args.cutout else ''] + ['aug' if args.auto_aug else ''])
    if aug_str[0] != '_': aug_str = '_' + aug_str
    if aug_str[-1] != '_': aug_str = aug_str + '-'
    model_name += args.dataset.lower() + aug_str + 'snn' + '_t' + str(
        args.T) + '_' + args.arch.lower() + '_act_' + args.act + '_width_' + str(
        args.alpha) + '-opt_' + args.optim.lower() + (
                          '_' + args.bn_type) + ('_bias' if args.bias else '') + '_wd_' + str(args.wd) + (
                      '_p_' + str(args.p) + '_' if args.p else '') + (
                      '_gamma_' + str(args.gamma) + '_' if args.gamma else '')
    cas_num = len([one for one in os.listdir(args.log_path) if one.startswith(model_name)])
    model_name += '_cas_' + str(cas_num)
    print('Generate custom_model name: ' + model_name)
    return model_name


if __name__ == '__main__':
    # set random seed, device, data type
    from config.config import args

    setup_seed(args.seed)
    dtype = torch.float
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # runtime configures
    args.log_interval = 100
    # practical config for learning scheduler
    model_name = get_model_name('', args)

    # redirect the output
    args.log_path = os.path.join(args.log_path, model_name)
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    log = Logger(args, args.log_path)
    log.info_config(args)
    # sys.stdout = Logger(os.path.join(args.log_path, model_name + '.log'))
    # sys.stderr = Logger(os.path.join(args.log_path, model_name + '.log'))
    main()
