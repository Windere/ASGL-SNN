# -*- coding: utf-8 -*-
import os
import argparse

parser = argparse.ArgumentParser(description='Training SNN')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--arch', default="VGG16", type=str, help="VGG5|VGG9|VGG11|VGG13|VGG16|"
                                                              "VGG19|CIFAR|ALEX")  # used
parser.add_argument('--dataset', default="CIFAR10", type=str, help="CIFAR10|CIFAR100|MNIST")  # used
parser.add_argument('--data_path', default="your data path", type=str)  # used
parser.add_argument('--ckpt_path', default="./checkpoint", type=str, help="checkpoint path")  # used
parser.add_argument('--log_path', default="./log", type=str, help="log path")  # used
parser.add_argument('--auto_aug', default=False, action='store_true')  # used
parser.add_argument('--cutout', default=False, action='store_true')  # used
parser.add_argument('--resume', default=None, type=str)  # used
parser.add_argument('--train_batch_size', default=128, type=int)  # used
parser.add_argument('--val_batch_size', default=128, type=int)  # used
parser.add_argument('--lr', default=0.1, type=float)  # used
parser.add_argument('--save_last', default=False, action='store_true')  # used
parser.add_argument('--bn_type', default='tdbn', type=str)  # used
parser.add_argument('--bias', default=False, action='store_true')  # used
parser.add_argument('--wd', default=5e-4, type=float)  # used
parser.add_argument('--num_epoch', default=300, type=int)  # used # check
parser.add_argument('--num_workers', default=16, type=int)  # used
parser.add_argument('--optim', default='SGDM', type=str)  # used
parser.add_argument('--grid_search', default='./summary.csv', type=str)  # used
parser.add_argument('--act', default='mns_sig', type=str)  # used # check
parser.add_argument('--alpha', default=1.0, type=float)  # used # check
parser.add_argument('--decay', default=None, type=float)  # used
parser.add_argument('--thresh', default=0.5, type=float)  # used
parser.add_argument('--p', default=None, type=float)  # used # check
parser.add_argument('--gamma', default=None, type=float)  # used # check
parser.add_argument('--train_thresh', action='store_true')  # used
parser.add_argument('--device', default='cuda:0', type=str)  # used
parser.add_argument('--T', default=2, type=int, help='num of time steps')  # used # check
parser.add_argument('--scheduler', default='COSINE', type=str)  # used
parser.add_argument('--lr_milestone', default=[0.3, 0.7, 0.9, 0.95], type=float, nargs='*', )  # used
parser.add_argument('--ns_milestone', default=[0, 0.2, 0.4, 0.6, 0.8, 0.95], type=float, nargs='*', )  # used

args = parser.parse_args()
# args = parser.parse_known_args()[0]
args.lr_milestone = [int(x * args.num_epoch) for x in args.lr_milestone]
if not os.path.exists(args.ckpt_path):
    os.mkdir(args.ckpt_path)
if not os.path.exists(args.log_path):
    os.mkdir(args.log_path)