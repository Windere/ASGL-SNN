# -*- coding: utf-8 -*-

import os
import argparse

parser = argparse.ArgumentParser(description='Training SNN')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--arch', default="VGGSNN2", type=str, help="VGG5|VGG9|VGG11|VGG13|VGG16|"
                                                              "VGG19|CIFAR|ALEX")  # used
parser.add_argument('--dataset', default="CIFAR10_DVS_Aug", type=str, help="dataset")  # used
parser.add_argument('--data_path', default="/data2/wzm/cifar10-dvs/events-pt", type=str)  # used
parser.add_argument('--ckpt_path', default="./checkpoint", type=str, help="checkpoint path")  # used
parser.add_argument('--log_path', default="./log", type=str, help="log path")  # used
parser.add_argument('--auto_aug', default=False, action='store_true')  # used
parser.add_argument('--cutout', default=False, action='store_true')  # used
parser.add_argument('--resume', default=None, type=str)  # used
parser.add_argument('--train_batch_size', default=32, type=int)  # used
parser.add_argument('--val_batch_size', default=32, type=int)  # used
parser.add_argument('--lr', default=0.1, type=float)  # used
parser.add_argument('--width_lr', default=3e-4, type=float)  # used
parser.add_argument('--save_last', default=False, action='store_true')  # used
parser.add_argument('--bn_type', default='', type=str)  # used
parser.add_argument('--bias', default=False, action='store_true')  # used
parser.add_argument('--wd', default=5e-4, type=float)  # used
parser.add_argument('--num_epoch', default=100, type=int)  # used # check
parser.add_argument('--num_workers', default=8, type=int)  # used
parser.add_argument('--optim', default='SGDM', type=str)  # used
parser.add_argument('--grid_search', default='./summary.csv', type=str)  # used
parser.add_argument('--act', default='spike', type=str)  # used # check
parser.add_argument('--alpha', default=1.0, type=float)  # used # check
parser.add_argument('--use_gate', default=False, action='store_true')  # used # check
# parser.add_argument('--neuron_wise', default=False, action='store_true')  # used # check

parser.add_argument('--granularity', default='layer',  type=str)  # used # check

parser.add_argument('--decay', default=None, type=float)  # used
parser.add_argument('--thresh', default=0.5, type=float)  # used
parser.add_argument('--p', default=None, type=float)  # used # check
parser.add_argument('--gamma', default=None, type=float)  # used # check
parser.add_argument('--train_decay', action='store_true')  # used
parser.add_argument('--train_thresh', action='store_true')  # used
parser.add_argument('--device', default='cuda:0', type=str)  # used
parser.add_argument('--T', default=10, type=int, help='num of time steps')  # used # check
parser.add_argument('--scheduler', default='COSINE', type=str)  # used
parser.add_argument('--lr_milestone', default=[0.3, 0.7, 0.9, 0.95], type=float, nargs='*', )  # used
parser.add_argument('--ns_milestone', default=[0, 0.2, 0.4, 0.6, 0.8, 0.95], type=float, nargs='*', )  # used


parser.add_argument('--means', default=1.0, type=float, metavar='N',
                    help='make all the potential increment around the means (default: 1.0)')
parser.add_argument('--lamb', default=1e-3, type=float, metavar='N',
                    help='adjust the norm factor to avoid outlier (default: 0.0)')
# parser.add_argument('--train_radio', default=0.9, type=float)  # used

parser.add_argument('--train_width', action='store_true')  # used

args = parser.parse_args()
# args = parser.parse_known_args()[0]
args.lr_milestone = [int(x * args.num_epoch) for x in args.lr_milestone]
if not os.path.exists(args.ckpt_path):
    os.mkdir(args.ckpt_path)
if not os.path.exists(args.log_path):
    os.mkdir(args.log_path)
print(args)
if not args.data_path.startswith('.') and not args.data_path.startswith('/'):
    path_list = args.data_path.split('/')
    root = os.environ.get(path_list[0])
    args.data_path = os.path.join(root, os.path.join(*path_list[1:]))
print(args.data_path)
defaults = vars(parser.parse_args([]))
args_in = vars(args)
for key in args_in:
    if args_in[key].__hash__ is None or not isinstance(args_in[key].__hash__(), int):
        args_in[key] = tuple(args_in[key])
# args.lr_interval = tuple(args.lr_interval)
# defaults.lr_interval = tuple(defaults.lr_interval)
for key in defaults:
    if defaults[key].__hash__ is None or not isinstance(defaults[key].__hash__(), int):
        defaults[key] = tuple(defaults[key])
args.cmd = [' --' + str(k).strip() + ' ' + str(v).strip() for k, v in set(args_in.items()) - set(defaults.items())]
