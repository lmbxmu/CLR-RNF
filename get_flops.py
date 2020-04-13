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

from thop import profile

from utils.common import graph_weight, kmeans_weight, random_weight, getloss,random_project, direct_project

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()

# Load pretrained model
print('==> Loading pretrained model..')
if args.pretrain_model is None or not os.path.exists(args.pretrain_model):
    raise ('Pretrained_model path should be exist!')
ckpt = torch.load(args.pretrain_model, map_location=device)
if args.arch == 'resnet_imagenet':
    origin_model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    origin_model.load_state_dict(ckpt)
elif args.arch == 'mobilenet_v1':
    origin_model = import_module(f'model.{args.arch}').mobilenet_v1().to(device)
    origin_model.load_state_dict(ckpt['state_dict'])
elif args.arch == 'mobilenet_v2':
    origin_model = import_module(f'model.{args.arch}').mobilenet_v2().to(device)
    origin_model.load_state_dict(ckpt['state_dict'])
elif args.arch == 'vgg_cifar':
    origin_model = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)
    origin_model.load_state_dict(ckpt['state_dict'])
elif args.arch == 'resnet_cifar':
    origin_model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    origin_model.load_state_dict(ckpt['state_dict'])
elif args.arch == 'googlenet':
    origin_model = import_module(f'model.{args.arch}').googlenet().to(device)
    origin_model.load_state_dict(ckpt['state_dict'])
else:
    raise('arch not exist!')



def graph_vgg(pr_target):    

    pr_cfg = []
    weights = []

    cfg = []
    centroids_state_dict = {}
    prune_state_dict = []
    indices = []

    current_layer = 0

    #Sort the weights and get the pruning threshold
    for name, module in origin_model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_weight = module.weight.data
            weights.append(conv_weight.view(-1))

    all_weights = torch.cat(weights,0)
    preserve_num = int(all_weights.size(0) * (1-pr_target))
    preserve_weight, _ = torch.topk(torch.abs(all_weights), preserve_num)
    threshold = preserve_weight[preserve_num-1]

    #Based on the pruning threshold, the prune cfg of each layer is obtained
    for weight in weights:
        pr_cfg.append(torch.sum(torch.lt(torch.abs(weight),threshold)).item()/weight.size(0))

    #Get the preseverd filters after pruning by graph method based on pruning proportion
    for name, module in origin_model.named_modules():

        if isinstance(module, nn.Conv2d):

            conv_weight = module.weight.data
            _, _, centroids, indice = graph_weight(conv_weight, int(conv_weight.size(0) * (1 - pr_cfg[current_layer])),logger)

            cfg.append(len(centroids))
            indices.append(indice)
            centroids_state_dict[name + '.weight'] = centroids.reshape((-1, conv_weight.size(1), conv_weight.size(2), conv_weight.size(3)))
            prune_state_dict.append(name + '.bias')
            current_layer+=1

        elif isinstance(module, nn.BatchNorm2d):
            prune_state_dict.append(name + '.weight')
            prune_state_dict.append(name + '.bias')
            prune_state_dict.append(name + '.running_var')
            prune_state_dict.append(name + '.running_mean')

        elif isinstance(module, nn.Linear):
            prune_state_dict.append(name + '.weight')
            prune_state_dict.append(name + '.bias')

    model = import_module(f'model.{args.arch}').VGG(args.cfg, layer_cfg=cfg).to(device)
    return model


def graph_resnet(pr_target):

    pr_cfg = []
    weights = []

    cfg = []
    centroids_state_dict = {}
    prune_state_dict = []

    current_block = 0

    #Sort the weights and get the pruning threshold
    for name, module in origin_model.named_modules():
        if isinstance(module, ResBasicBlock):

            conv_weight = module.conv1.weight.data
            weights.append(conv_weight.view(-1))

    all_weights = torch.cat(weights,0)
    preserve_num = int(all_weights.size(0) * (1-pr_target))
    preserve_weight, _ = torch.topk(torch.abs(all_weights), preserve_num)
    threshold = preserve_weight[preserve_num-1]

    #Based on the pruning threshold, the prune cfg of each layer is obtained
    for weight in weights:
        pr_cfg.append(torch.sum(torch.lt(torch.abs(weight),threshold)).item()/weight.size(0))
    
    #Get the preseverd filters after pruning by graph method based on pruning proportion

    for name, module in origin_model.named_modules():

        if isinstance(module, ResBasicBlock):

            conv1_weight = module.conv1.weight.data

            _, _, centroids, indice = graph_weight(conv1_weight, int(conv1_weight.size(0) * (1 - pr_cfg[current_block])),logger)
            cfg.append(len(centroids))
            centroids_state_dict[name + '.conv1.weight'] = centroids
            if args.init_method == 'random_project':
                centroids_state_dict[name + '.conv2.weight'] = random_project(module.conv2.weight.data, len(centroids))
            else:
                centroids_state_dict[name + '.conv2.weight'] = direct_project(module.conv2.weight.data, indice)

            prune_state_dict.append(name + '.bn1.weight')
            prune_state_dict.append(name + '.bn1.bias')
            prune_state_dict.append(name + '.bn1.running_var')
            prune_state_dict.append(name + '.bn1.running_mean')

            current_block+=1


    model = import_module(f'model.{args.arch}').resnet(args.cfg, layer_cfg=cfg).to(device)
    return model


def graph_googlenet(pr_target):

    pr_cfg = []
    weights = []

    cfg = []
    centroids_state_dict = {}
    prune_state_dict = []
    indices = []

    current_index = 0

    for name, module in origin_model.named_modules():

        if isinstance(module, Inception):

            branch3_weight = module.branch3x3[0].weight.data
            branch5_weight1 = module.branch5x5[0].weight.data
            branch5_weight2 = module.branch5x5[3].weight.data
            weights.append(branch3_weight.view(-1))
            weights.append(branch5_weight1.view(-1))
            weights.append(branch5_weight2.view(-1))

    all_weights = torch.cat(weights,0)
    preserve_num = int(all_weights.size(0) * (1-pr_target))
    preserve_weight, _ = torch.topk(torch.abs(all_weights), preserve_num)
    threshold = preserve_weight[preserve_num-1]

    #Based on the pruning threshold, the prune cfg of each layer is obtained
    for weight in weights:
        pr_cfg.append(torch.sum(torch.lt(torch.abs(weight),threshold)).item()/weight.size(0))
    #Get the preseverd filters after pruning by graph method based on pruning proportion
    for name, module in origin_model.named_modules():

        if isinstance(module, Inception):

            branch3_weight = module.branch3x3[0].weight.data
            _, _, centroids, indice = graph_weight(branch3_weight, int(branch3_weight.size(0) * (1 - pr_cfg[current_index])),logger)
            cfg.append(len(centroids))
            centroids_state_dict[name + '.branch3x3.0.weight'] = centroids
            if args.init_method == 'random_project':
                centroids_state_dict[name + '.branch3x3.3.weight'] = random_project(module.branch3x3[3].weight.data, len(centroids))
            else:
                centroids_state_dict[name + '.branch3x3.3.weight'] = direct_project(module.branch3x3[3].weight.data, indice)

            prune_state_dict.append(name + '.branch3x3.0.bias')
            prune_state_dict.append(name + '.branch3x3.1.weight')
            prune_state_dict.append(name + '.branch3x3.1.bias')
            prune_state_dict.append(name + '.branch3x3.1.running_var')
            prune_state_dict.append(name + '.branch3x3.1.running_mean')

            current_index+=1
            branch5_weight1 = module.branch5x5[0].weight.data
            _, _, centroids, indice = graph_weight(branch5_weight1, int(branch5_weight1.size(0) * (1 - pr_cfg[current_index])),logger)
            cfg.append(len(centroids))
            indices.append(indice)
            centroids_state_dict[name + '.branch5x5.0.weight'] = centroids

            prune_state_dict.append(name + '.branch5x5.0.bias')
            prune_state_dict.append(name + '.branch5x5.1.weight')
            prune_state_dict.append(name + '.branch5x5.1.bias')
            prune_state_dict.append(name + '.branch5x5.1.running_var')
            prune_state_dict.append(name + '.branch5x5.1.running_mean')

            current_index+=1
            branch5_weight2 = module.branch5x5[3].weight.data
            _, _, centroids, indice = graph_weight(branch5_weight2, int(branch5_weight2.size(0) * (1 - pr_cfg[current_index])),logger)
            cfg.append(len(centroids))
            centroids_state_dict[name + '.branch5x5.3.weight'] = centroids.reshape((-1, branch5_weight2.size(1), branch5_weight2.size(2), branch5_weight2.size(3)))

            if args.init_method == 'random_project':
                centroids_state_dict[name + '.branch5x5.6.weight'] = random_project(module.branch5x5[6].weight.data, len(centroids))
            else:
                centroids_state_dict[name + '.branch5x5.6.weight'] = direct_project(module.branch5x5[6].weight.data, indice)

            prune_state_dict.append(name + '.branch5x5.3.bias')
            prune_state_dict.append(name + '.branch5x5.4.weight')
            prune_state_dict.append(name + '.branch5x5.4.bias')
            prune_state_dict.append(name + '.branch5x5.4.running_var')
            prune_state_dict.append(name + '.branch5x5.4.running_mean')
    model = import_module(f'model.{args.arch}').googlenet(layer_cfg=cfg).to(device)
    return model


def graph_resnet_imagenet(pr_target):

    pr_cfg = []
    weights = []

    cfg = []
    centroids_state_dict = {}
    prune_state_dict = []
    indices = []

    current_index = 0
    #Sort the weights and get the pruning threshold
    for name, module in origin_model.named_modules():

        if isinstance(module, BasicBlock):
            conv1_weight = module.conv1.weight.data
            weights.append(conv1_weight.view(-1))

        elif isinstance(module, Bottleneck):

            conv1_weight = module.conv1.weight.data
            weights.append(conv1_weight.view(-1))
            conv2_weight = module.conv2.weight.data
            weights.append(conv2_weight.view(-1))
            
    all_weights = torch.cat(weights,0)
    preserve_num = int(all_weights.size(0) * (1-pr_target))
    preserve_weight, _ = torch.topk(torch.abs(all_weights), preserve_num)
    threshold = preserve_weight[preserve_num-1]

    #Based on the pruning threshold, the prune cfg of each layer is obtained
    for weight in weights:
        pr_cfg.append(torch.sum(torch.lt(torch.abs(weight),threshold)).item()/weight.size(0))
    #Get the preseverd filters after pruning by graph method based on pruning proportion
    for name, module in origin_model.named_modules():

        if isinstance(module, BasicBlock):

            conv1_weight = module.conv1.weight.data
            _, _, centroids, indice = graph_weight(conv1_weight, int(conv1_weight.size(0) * (1 - pr_cfg[current_index])),logger)
            cfg.append(len(centroids))
            cfg.append(0) #assume baseblock has three conv layer
            centroids_state_dict[name + '.conv1.weight'] = centroids
            if args.init_method == 'random_project':
                centroids_state_dict[name + '.conv2.weight'] = random_project(module.conv2.weight.data, len(centroids))
            else:
                centroids_state_dict[name + '.conv2.weight'] = direct_project(module.conv2.weight.data, indice)

            prune_state_dict.append(name + '.bn1.weight')
            prune_state_dict.append(name + '.bn1.bias')
            prune_state_dict.append(name + '.bn1.running_var')
            prune_state_dict.append(name + '.bn1.running_mean')
            current_index+=1

        elif isinstance(module, Bottleneck):

            conv1_weight = module.conv1.weight.data
            _, _, centroids, indice = graph_weight(conv1_weight, int(conv1_weight.size(0) * (1 - pr_cfg[current_index])),logger)
            cfg.append(len(centroids))
            indices.append(indice)
            centroids_state_dict[name + '.conv1.weight'] = centroids

            prune_state_dict.append(name + '.bn1.weight')
            prune_state_dict.append(name + '.bn1.bias')
            prune_state_dict.append(name + '.bn1.running_var')
            prune_state_dict.append(name + '.bn1.running_mean')
            current_index+=1

            conv2_weight = module.conv2.weight.data
            _, _, centroids, indice = graph_weight(conv2_weight, int(conv2_weight.size(0) * (1 - pr_cfg[current_index])),logger)
            cfg.append(len(centroids))
            centroids_state_dict[name + '.conv2.weight'] = centroids.reshape((-1, conv2_weight.size(1), conv2_weight.size(2), conv2_weight.size(3)))

            if args.init_method == 'random_project':
                centroids_state_dict[name + '.conv3.weight'] = random_project(module.conv3.weight.data, len(centroids))
            else:
                centroids_state_dict[name + '.conv3.weight'] = direct_project(module.conv3.weight.data, indice)

            prune_state_dict.append(name + '.bn2.weight')
            prune_state_dict.append(name + '.bn2.bias')
            prune_state_dict.append(name + '.bn2.running_var')
            prune_state_dict.append(name + '.bn2.running_mean')
            current_index+=1

    model = import_module(f'model.{args.arch}').resnet(args.cfg, layer_cfg=cfg).to(device)

    return model   

def graph_mobilenet_v1(pr_target):    

    pr_cfg = []
    weights = []

    cfg = []
    centroids_state_dict = {}
    prune_state_dict = []
    indices = []

    i = 0
    #Sort the weights and get the pruning threshold
    for name, module in origin_model.named_modules():
        if isinstance(module, nn.Conv2d):
            if i == 0:
                conv_weight = module.weight.data
                weights.append(conv_weight.view(-1))
            elif i % 2 == 1:
                conv_weight = module.weight.data
            else:
                conv_weight_1 = module.weight.data
                weights.append(torch.cat((conv_weight.view(-1),conv_weight_1.view(-1)),0))
            i += 1

    all_weights = torch.cat(weights,0)
    preserve_num = int(all_weights.size(0) * (1-pr_target))
    preserve_weight, _ = torch.topk(torch.abs(all_weights), preserve_num)
    threshold = preserve_weight[preserve_num-1]

    #Based on the pruning threshold, the prune cfg of each layer is obtained
    for weight in weights:
        pr_cfg.append(torch.sum(torch.lt(torch.abs(weight),threshold)).item()/weight.size(0))
    #print(pr_cfg)

    current_layer = 0

    flag = True
    #Get the preseverd filters after pruning by graph method based on pruning proportion
    for name, module in origin_model.named_modules():

        if isinstance(module, nn.Conv2d):

            conv_weight = module.weight.data
            #print(conv_weight.size())
            _, _, centroids, indice = graph_weight(conv_weight, int(conv_weight.size(0) * (1 - pr_cfg[current_layer])),logger)
            if flag:
                current_layer -= 1
                cfg.append(len(centroids))
            current_layer += 1
            flag = not flag


            indices.append(indice)
            centroids_state_dict[name + '.weight'] = centroids.reshape((-1, conv_weight.size(1), conv_weight.size(2), conv_weight.size(3)))
            prune_state_dict.append(name + '.bias')

        elif isinstance(module, nn.BatchNorm2d):
            prune_state_dict.append(name + '.weight')
            prune_state_dict.append(name + '.bias')
            prune_state_dict.append(name + '.running_var')
            prune_state_dict.append(name + '.running_mean')

        elif isinstance(module, nn.Linear):
            prune_state_dict.append(name + '.weight')
            prune_state_dict.append(name + '.bias')


    #load weight
    model = import_module(f'model.{args.arch}').mobilenet_v1(layer_cfg=cfg).to(device)

    return model

def main():
    print('==> Building Model..')

    if args.arch == 'vgg_cifar':
        model = graph_vgg(args.pr_target)
    elif args.arch == 'resnet_cifar':
        model = graph_resnet(args.pr_target)
    elif args.arch == 'googlenet':
        model = graph_googlenet(args.pr_target)
    elif args.arch == 'resnet_imagenet':
        model = graph_resnet_imagenet(args.pr_target)
    else:
        raise('arch not exist!')
    print("Graph Down!")
    orichannel = 0
    channel = 0

    #Calculate the flops and params of origin_model & pruned_model
    Input = torch.randn(1, 3, 224, 224).to(device)
    oriflops, oriparams = profile(origin_model, inputs=(Input, ))
    flops, params = profile(model, inputs=(Input, ))

    for name, module in origin_model.named_modules():
        if isinstance(module, nn.Conv2d):
            orichannel += origin_model.state_dict()[name + '.weight'].size(0)
            #print(orimodel.state_dict()[name + '.weight'].size(0))

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            channel += model.state_dict()[name + '.weight'].size(0)
            #print(model.state_dict()[name + '.weight'].size(0))

    logger.info('--------------UnPrune Model--------------')
    logger.info('Channels: %d'%(orichannel))
    logger.info('Params: %.2f M '%(oriparams/1000000))
    logger.info('FLOPS: %.2f M '%(oriflops/1000000))

    logger.info('--------------Prune Model--------------')
    logger.info('Channels:%d'%(channel))
    logger.info('Params: %.2f M'%(params/1000000))
    logger.info('FLOPS: %.2f M'%(flops/1000000))


    logger.info('--------------Compress Rate--------------')
    logger.info('Channels Prune Rate: %d/%d (%.2f%%)' % (channel, orichannel, 100. * (orichannel - channel) / orichannel))
    logger.info('Params Compress Rate: %.2f M/%.2f M(%.2f%%)' % (params/1000000, oriparams/1000000, 100. * (oriparams- params) / oriparams))
    logger.info('FLOPS Compress Rate: %.2f M/%.2f M(%.2f%%)' % (flops/1000000, oriflops/1000000, 100. * (oriflops- flops) / oriflops))



if __name__ == '__main__':
    main()