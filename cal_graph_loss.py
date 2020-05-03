import torch
import torch.nn as nn
import torch.optim as optim
from utils.options import args
from model.resnet_cifar import ResBasicBlock
from model.googlenet import Inception
from model.resnet_imagenet import BasicBlock, Bottleneck
import utils.common as utils

import os
import time
from importlib import import_module

from utils.common import graph_weight, kmeans_weight, random_weight, getloss

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()


def graph_vgg(pr_target):

    ckpt = torch.load(args.pretrain_model, map_location=device)
    origin_model = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)
    origin_model.load_state_dict(ckpt['state_dict'])

    indices = []
    current_layer = 0

    for name, module in origin_model.named_modules():

        if isinstance(module, nn.Conv2d):

            conv_weight = module.weight.data
            
            m = int(conv_weight.size(0) * (1 - pr_target))#Number of channels to keep

            
            #s_matrix is the origin similarity matrix, m_..matrix is the NXM similarity matrix
            m_knn, s_matrix, _, _ = graph_weight(conv_weight, m, logger)
            m_kmeans,_,_ = kmeans_weight(conv_weight, m)
            m_random,_,_ = random_weight(conv_weight, m)

            loss_knn = getloss(m_knn, s_matrix)
            loss_kmeans = getloss(m_kmeans, s_matrix)
            loss_random = getloss(m_random, s_matrix)

            logger.info('layer[{}]\t'
                'loss_knn:{:.6f}\t'
                'loss_kmeans:{:.6f}\t'
                'loss_random:{:.6f}'.format(current_layer,loss_knn,loss_kmeans,loss_random)
                )
            
            current_layer+=1

def graph_resnet(pr_target):

    ckpt = torch.load(args.pretrain_model, map_location=device)
    origin_model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    origin_model.load_state_dict(ckpt['state_dict'])

    cfg = []
    centroids_state_dict = {}
    prune_state_dict = []

    current_block = 0

    for name, module in origin_model.named_modules():

        if isinstance(module, ResBasicBlock):

            conv_weight = module.conv1.weight.data

            m = int(conv_weight.size(0) * (1 - pr_target))#Number of channels to keep

            #s_matrix is the origin similarity matrix, m_..matrix is the NXM similarity matrix
            m_knn, s_matrix, _, _ = graph_weight(conv_weight, m, logger)
            m_kmeans, _, _ = kmeans_weight(conv_weight, m)
            m_random, _, _= random_weight(conv_weight, m)

            loss_knn = getloss(m_knn, s_matrix)
            loss_kmeans = getloss(m_kmeans, s_matrix)
            loss_random = getloss(m_random, s_matrix)

            logger.info('Block[{}]\t'
                'loss_knn:{:.2f}\t'
                'loss_kmeans:{:.2f}\t'
                'loss_random:{:.2f}'.format(current_block,loss_knn,loss_kmeans,loss_random)
                )
            current_block+=1

def graph_resnet_imagenet(pr_target):

    ckpt = torch.load(args.pretrain_model, map_location=device)
    origin_model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    origin_model.load_state_dict(ckpt)

    cfg = []
    centroids_state_dict = {}
    prune_state_dict = []
    indices = []

    current_block = 0

    for name, module in origin_model.named_modules():

        if isinstance(module, BasicBlock):

            conv1_weight = module.conv1.weight.data

            m = int(conv1_weight.size(0) * (1 - pr_target))#Number of channels to keep

            #s_matrix is the origin similarity matrix, m_..matrix is the NXM similarity matrix
            m_knn, s_matrix, _, _ = graph_weight(conv1_weight, m, logger)
            m_kmeans, _, _ = kmeans_weight(conv1_weight, m)
            m_random, _, _ = random_weight(conv1_weight, m)

            loss_knn = getloss(m_knn, s_matrix)
            loss_kmeans = getloss(m_kmeans, s_matrix)
            loss_random = getloss(m_random, s_matrix)

            logger.info('Block[{}]\t'
                'loss_knn:{:.2f}\t'
                'loss_kmeans:{:.2f}\t'
                'loss_random:{:.2f}'.format(current_block,loss_knn,loss_kmeans,loss_random)
                )
            current_block += 1


        elif isinstance(module, Bottleneck):

            conv1_weight = module.conv1.weight.data

            m = int(conv1_weight.size(0) * (1 - pr_target))#Number of channels to keep

            #s_matrix is the origin similarity matrix, m_..matrix is the NXM similarity matrix
            m_knn, s_matrix, _, _ = graph_weight(conv1_weight, m, logger)
            m_kmeans, _, _ = kmeans_weight(conv1_weight, m)
            m_random, _, _ = random_weight(conv1_weight, m)

            loss_knn = getloss(m_knn, s_matrix)
            loss_kmeans = getloss(m_kmeans, s_matrix)
            loss_random = getloss(m_random, s_matrix)

            logger.info('Block[{}]\tconv1\t'
                'loss_knn:{:.2f}\t'
                'loss_kmeans:{:.2f}\t'
                'loss_random:{:.2f}'.format(current_block,loss_knn,loss_kmeans,loss_random)
                )
            

            conv2_weight = module.conv2.weight.data

            m = int(conv2_weight.size(0) * (1 - pr_target))#Number of channels to keep

            #s_matrix is the origin similarity matrix, m_..matrix is the NXM similarity matrix
            m_knn, s_matrix, _, _ = graph_weight(conv2_weight, m, logger)
            m_kmeans, _, _ = kmeans_weight(conv2_weight, m)
            m_random, _, _ = random_weight(conv2_weight, m)

            loss_knn = getloss(m_knn, s_matrix)
            loss_kmeans = getloss(m_kmeans, s_matrix)
            loss_random = getloss(m_random, s_matrix)

            logger.info('Block[{}]\tconv2\t'
                'loss_knn:{:.2f}\t'
                'loss_kmeans:{:.2f}\t'
                'loss_random:{:.2f}'.format(current_block,loss_knn,loss_kmeans,loss_random)
                )
            current_block += 1

def graph_googlenet(pr_target):

    ckpt = torch.load(args.pretrain_model, map_location=device)
    origin_model = import_module(f'model.{args.arch}').googlenet().to(device)
    origin_model.load_state_dict(ckpt['state_dict'])

    cfg = []
    centroids_state_dict = {}
    prune_state_dict = []
    indices = []
    current_inception = 0

    for name, module in origin_model.named_modules():

        if isinstance(module, Inception):

            branch3_weight = module.branch3x3[0].weight.data

            m = int(branch3_weight.size(0) * (1 - pr_target))#Number of channels to keep

            #s_matrix is the origin similarity matrix, m_..matrix is the NXM similarity matrix
            m_knn, s_matrix, _, _ = graph_weight(branch3_weight, m, logger)
            m_kmeans, _, _ = kmeans_weight(branch3_weight, m)
            m_random, _, _ = random_weight(branch3_weight, m)

            loss_knn = getloss(m_knn, s_matrix)
            loss_kmeans = getloss(m_kmeans, s_matrix)
            loss_random = getloss(m_random, s_matrix)

            logger.info('Inception[{}]\tBranch[3]\t'
                'loss_knn:{:.2f}\t'
                'loss_kmeans:{:.2f}\t'
                'loss_random:{:.2f}'.format(current_inception,loss_knn,loss_kmeans,loss_random)
                )

            branch5_weight1 = module.branch5x5[0].weight.data

            m = int(branch5_weight1.size(0) * (1 - pr_target))#Number of channels to keep

            #s_matrix is the origin similarity matrix, m_..matrix is the NXM similarity matrix
            m_knn, s_matrix, _, _ = graph_weight(branch5_weight1, m, logger)
            m_kmeans, _, _ = kmeans_weight(branch5_weight1, m)
            m_random, _, _ = random_weight(branch5_weight1, m)

            loss_knn = getloss(m_knn, s_matrix)
            loss_kmeans = getloss(m_kmeans, s_matrix)
            loss_random = getloss(m_random, s_matrix)

            logger.info('Inception[{}]\tBranch[5.1]\t'
                'loss_knn:{:.2f}\t'
                'loss_kmeans:{:.2f}\t'
                'loss_random:{:.2f}'.format(current_inception,loss_knn,loss_kmeans,loss_random)
                )

            branch5_weight2 = module.branch5x5[3].weight.data

            m = int(branch5_weight2.size(0) * (1 - pr_target))#Number of channels to keep

            #s_matrix is the origin similarity matrix, m_..matrix is the NXM similarity matrix
            m_knn, s_matrix, _, _ = graph_weight(branch5_weight2, m, logger)
            m_kmeans, _, _ = kmeans_weight(branch5_weight2, m)
            m_random, _, _ = random_weight(branch5_weight2, m)

            loss_knn = getloss(m_knn, s_matrix)
            loss_kmeans = getloss(m_kmeans, s_matrix)
            loss_random = getloss(m_random, s_matrix)

            logger.info('Inception[{}]\tBranch[5.2]\t'
                'loss_knn:{:.2f}\t'
                'loss_kmeans:{:.2f}\t'
                'loss_random:{:.2f}'.format(current_inception,loss_knn,loss_kmeans,loss_random)
                )
            current_inception += 1
            
      

def main():
    print('==> Building Model..')
    pr_rate = 0.5
    if args.pretrain_model is None or not os.path.exists(args.pretrain_model):
        raise ('Pretrained_model path should be exist!')
    if args.arch == 'vgg_cifar':
        graph_vgg(pr_rate)
    elif args.arch == 'resnet_cifar':
        graph_resnet(pr_rate)
    elif args.arch == 'googlenet':
        graph_googlenet(pr_rate)
    elif args.arch == 'resnet_imagenet':
        graph_resnet_imagenet(pr_rate)
    else:
        raise('arch not exist!')
    print("Graph Down!")


if __name__ == '__main__':
    main()