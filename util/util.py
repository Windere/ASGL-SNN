import torch
import numpy as np
import random
import torch.nn as nn
import sys
import copy
import logging
import os


class Logger:
    def __init__(self, args, log_path, write_file=True):
        self.log_path = log_path
        self.logger = logging.getLogger('')
        if write_file:
            filename = os.path.join(self.log_path, 'train.log')
            # file handler
            handler = logging.FileHandler(filename=filename, mode="w")
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))

        # console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter('%(message)s'))

        self.logger.setLevel(logging.INFO)
        if write_file:
            self.logger.addHandler(handler)
            self.logger.info("Logger created at {}".format(filename))
        self.logger.addHandler(console)
        # filename = os.path.join(self.log_path, 'test.log')
        # # file handler
        # handler = logging.FileHandler(filename=filename, mode="w")
        # handler.setLevel(logging.INFO)
        # handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))

        # # console handler
        # console = logging.StreamHandler()
        # console.setLevel(logging.INFO)
        # console.setFormatter(logging.Formatter('%(message)s'))

        # self.logger.setLevel(logging.INFO)
        # self.logger.addHandler(handler)
        # self.logger.addHandler(console)
        # self.logger.info("Logger created at {}".format(filename))

    def debug(self, strout):
        return self.logger.debug(strout)

    def info(self, strout):
        return self.logger.info(strout)

    def info_config(self, config):
        self.info('The hyperparameter list:')
        for k, v in vars(config).items():
            self.info('  --' + k + ' ' + str(v))


def setup_seed(seed):
    import os
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    print('set random seed as ' + str(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quantile(tensor: torch.Tensor, p: float):
    try:
        return torch.quantile(tensor, p)
    except:
        tensor_np = tensor.cpu().detach().numpy()
        return torch.tensor(np.percentile(tensor_np, q=p * 100)).type_as(tensor)
