from __future__ import absolute_import
import datetime
import shutil
from pathlib import Path
import logging
import torch
import torch.nn as nn
from .options import args
import numpy as np
import math
import torch.nn.functional as F
import random
from sklearn.cluster import KMeans
from sklearn.random_projection import SparseRandomProjection

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



def graph_weight(weight,m,logger):

    if args.graph_gpu:
        W = weight.clone() 
    else:
        W = weight.cpu().clone()
    if weight.dim() == 4:  #Convolution layer
        W = W.view(W.size(0), -1)
    else:
        raise('The weight dim must be 4!')
    device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
    #Calculate the similarity matrix and normalize
    s_matrix = F.normalize(torch.exp(-pairwise_distances(W)),1)
    #Sort
    sorted_value, indices = torch.sort(s_matrix,descending=True)
   
    if args.graph_gpu:
        indices = indices.cpu()
    indices = indices.numpy()

    k = m #Calculate the nearest k channels of each channel
    m_tmp = 0 #Common nearest channels after k nearest neighbor channels intersect
    while m_tmp < m:
        #Take the nearest k filters
        indicek = indices[:,:k].tolist()
        #Intersect k nearest neighbors of all filters
        indicek = set(indicek[0]).intersection(*indicek[1:])
        m_tmp = len(indicek)
        if m_tmp > m:
            #Take the difference set for the result of the last KNN, 
            #and randomly select the filter from the difference set until the target m is reached
            pre_indicek = indices[:,:k-1].tolist()
            pre_indicek = set(pre_indicek[0]).intersection(*pre_indicek[1:])
            redundant_indice = indicek.difference(pre_indicek)
            while len(pre_indicek) != m:
                pre_indicek.add(redundant_indice.pop())
            indicek = pre_indicek
            m_tmp = m

        #logger.info('k[{}]\tm_tmp[{}]\ttarget m[{}]'.format(k,m_tmp,m))
        k += 1
    if args.graph_gpu:
        Wprune = torch.index_select(W,0,torch.tensor(list(indicek)).to(device))
    else:
        Wprune = torch.index_select(W,0,torch.tensor(list(indicek)))
    m_matrix = F.normalize(torch.exp(-pairwise_distances(W,Wprune)),1)
    Wprune = Wprune.cpu()
    return m_matrix, s_matrix, Wprune, indicek

def kmeans_weight(weight,m):
    W = weight.cpu().clone()
    if weight.dim() == 4:  #Convolution layer
        W = W.view(W.size(0), -1)
    else:
        raise('The weight dim must be 4!')
    device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

    kmeans = KMeans(n_clusters=m, random_state=0).fit(W.numpy())
    if args.graph_gpu:
        m_matrix = F.normalize(torch.exp(-pairwise_distances(W.to(device),torch.from_numpy(kmeans.cluster_centers_).to(device))),1)
    else:
        m_matrix = F.normalize(torch.exp(-pairwise_distances(W,torch.from_numpy(kmeans.cluster_centers_))),1)
    return m_matrix


def random_weight(weight,m):
    
    W = weight.cpu().clone()
    if weight.dim() == 4:  #Convolution layer
        W = W.view(W.size(0), -1)
    else:
        raise('The weight dim must be 4!')
    device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
    indices = random.sample(range(0, W.size(0)-1), m)
    if args.graph_gpu:
        W = W.to(device)
        Wprune = torch.index_select(W,0,torch.tensor(list(indices)).to(device))
    else:
        Wprune = torch.index_select(W,0,torch.tensor(list(indices)))
    m_matrix = F.normalize(torch.exp(-pairwise_distances(W,Wprune)),1)
    return m_matrix

def getloss(B,A):
    #loss = torch.pow(torch.norm(A - torch.mm(B, B.t()),2),2)
    loss = torch.norm(A - torch.mm(B, B.t()),1)/(A.size(0)*A.size(0))
    '''
    f_num = A.size(0)
    A_1 = torch.zeros(f_num,f_num)
    for i in range(f_num):
        for j in range(f_num):
            A_1[i,j] = torch.sum(torch.mul(B[i,:],B[j,:]))
    loss = torch.pow(torch.norm(A - A_1,1),1)
    '''
    return loss


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


def random_project(weight, channel_num):

    A = weight.cpu().clone()
    A = A.view(A.size(0), -1)
    rp = SparseRandomProjection(n_components=channel_num * weight.size(2) * weight.size(3))
    rp.fit(A)
    return rp.transform(A)

def direct_project(weight, indices):
    #print(weight.size())

    A = torch.randn(weight.size(0), len(indices), weight.size(2), weight.size(3))
    #print(A.size())
    for i, indice in enumerate(indices):

        A[:, i, :, :] = weight[:, indice, :, :]

    return A

#label smooth
class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss

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