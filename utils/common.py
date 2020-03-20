from __future__ import absolute_import
import datetime
import shutil
from pathlib import Path
import logging
import torch
from .options import args
import numpy as np
import math
import torch.nn.functional as F

import os

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class checkpoint():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        self.args = args
        self.job_dir = Path(args.job_dir)
        self.ckpt_dir = self.job_dir / 'checkpoint'
        self.run_dir = self.job_dir / 'run'

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.job_dir)
        _make_dir(self.ckpt_dir)
        _make_dir(self.run_dir)

        config_dir = self.job_dir / 'config.txt'
        with open(config_dir, 'w') as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save_model(self, state, epoch, is_best):
        save_path = f'{self.ckpt_dir}/model_last.pt'
        torch.save(state, save_path)
        if is_best:
            shutil.copyfile(save_path, f'{self.ckpt_dir}/model_best.pt')

def graph_weight(weight,k):

    if args.graph_gpu:
        W = weight.clone() 
    else:
        W = weight.cpu().clone()
    f_num = W.size(0)
    if weight.dim() == 4:  #Convolution layer
        W = W.view(f_num, -1)
    else:
        raise('The weight dim must be 4!')

    s_matrix = torch.eye(f_num)

    #Calculate the similarity matrix
    for i in range(f_num):
        s_matrix[i] = F.pairwise_distance(torch.unsqueeze(W[i],0).repeat(f_num,1),W)

    s_matrix = torch.exp(-torch.pow(s_matrix,2))

    #First Sort
    sorted_value, _ = torch.sort(s_matrix,descending=True)

    #For each filter, divide the distance of the k nearest filter to it, 
    #that is, normalize according to the formula of graph hashing
    for i in range(f_num):
        s_matrix[i] = torch.div(s_matrix[i],torch.sum(sorted_value[i][1:k]))

    #Resort normalized distances
    _, indices = torch.sort(s_matrix,descending=True)

    #Take the nearest k filters
    indices = indices[:,:k].numpy()

    #Intersect k nearest neighbors of all filters
    indice = indices[0,:]
    for i in range(f_num-1):
        #print(indice)
        indice = list(set(indice).intersection(set(indices[i+1,:])))

    m = len(indice)

    return m, indice


def get_logger(file_path):
    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res