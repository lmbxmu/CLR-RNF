import torch
import torch.nn as nn
import torch.optim as optim
from utils.options import args
import utils.common as utils

import os
import time
import math
from data import imagenet_dali
from importlib import import_module
from model.resnet_imagenet import BasicBlock, Bottleneck
from model.mobilenet_v2 import InvertedResidual
from utils.common import *

from thop import profile
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.cuda()

# Data
print('==> Preparing data..')
def get_data_set(type='train'):
    if type == 'train':
        return imagenet_dali.get_imagenet_iter_dali('train', args.data_path, args.train_batch_size,
                                                   num_threads=4, crop=224, device_id=args.gpus[0], num_gpus=1)
    else:
        return imagenet_dali.get_imagenet_iter_dali('val', args.data_path, args.eval_batch_size,
                                                   num_threads=4, crop=224, device_id=args.gpus[0], num_gpus=1)
trainLoader = get_data_set('train')
testLoader = get_data_set('test')

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
    origin_model = import_module(f'model.{args.arch}_flops').mobilenet_v2().to(device)
else:
    raise('arch not exist!')

flops_cfg = {
    'resnet50':[2]*3+[1.5]*4+[1]*6+[0.5]*3,
    'mobilenet_v2':[1,3,1.5,0.5,2,1.5,1,0.5]
    #'resnet50':[1.0, 1.67262, 0.3869, 1.26786, 0.3869, 1.26786, 0.77381, 2.02679, 0.38393, 1.25298, 0.38393, 1.25298, 0.38393, 1.25298, 0.76786, 2.01339, 0.38244, 1.24554, 0.38244, 1.24554, 0.38244, 1.24554, 0.38244, 1.24554, 0.38244, 1.24554, 0.76488, 2.0067, 0.3817, 1.24182, 0.3817, 1.24182]
}
flops_lambda = {
 'resnet50':0.4,
  'mobilenet_v2':1
}


def graph_resnet(pr_target):

    pr_cfg = []
    block_cfg = []
    weights = []

    cfg = []
    centroids_state_dict = {}
    prune_state_dict = []
    indices = []

    current_index = 0
    index = 0
    #start_time = time.time()
    #Sort the weights and get the pruning threshold
    for name, module in origin_model.named_modules():

        if isinstance(module, BasicBlock):
            conv1_weight = torch.div(module.conv1.weight.data,math.pow(flops_cfg[args.cfg][index],flops_lambda[args.cfg]))
            #conv1_weight = module.conv1.weight.data
            weights.append(conv1_weight.view(-1))
            index += 1

        elif isinstance(module, Bottleneck):

            conv1_weight = torch.div(module.conv1.weight.data,math.pow(flops_cfg[args.cfg][index],flops_lambda[args.cfg]))
            #conv1_weight = module.conv1.weight.data
            weights.append(conv1_weight.view(-1))
            #conv2_weight = module.conv2.weight.data
            conv2_weight = torch.div(module.conv2.weight.data,math.pow(flops_cfg[args.cfg][index],flops_lambda[args.cfg]))
            weights.append(conv2_weight.view(-1))

            index += 1

    #for each stage, give the same prune rate for all blocks' output
    blocks_num = [3, 4, 6, 3]
    tmp_weight = []
    block_weights = []
    stage = 0
    block_index = 0
    index = 0
    for name, module in origin_model.named_modules():

        if isinstance(module, Bottleneck):

            conv_weight = torch.div(module.conv3.weight.data,math.pow(flops_cfg[args.cfg][index],flops_lambda[args.cfg]))
            #conv1_weight = module.conv1.weight.data
            if block_index == 0:
                tmp_weight = conv_weight.view(-1)
            else:
                tmp_weight = torch.cat((tmp_weight, conv_weight.view(-1)),0)
            block_index += 1
            if block_index == blocks_num[stage]:
                block_weights.append(tmp_weight)
                block_index = 0
                stage += 1
            index += 1
    weights.extend(block_weights)
    all_weights = torch.cat(weights,0)
    preserve_num = int(all_weights.size(0) * (1-pr_target))
    preserve_weight, _ = torch.topk(torch.abs(all_weights), preserve_num)
    threshold = preserve_weight[preserve_num-1]

    #Based on the pruning threshold, the prune cfg of each layer is obtained
    for weight in weights:
        pr_cfg.append(torch.sum(torch.lt(torch.abs(weight),threshold)).item()/weight.size(0))
    for weight in block_weights:
        block_cfg.append(torch.sum(torch.lt(torch.abs(weight),threshold)).item()/weight.size(0))
    #print(pr_cfg)
    #print(block_cfg)
    #current_time = time.time()
    #print("Find Structure Time {:.2f}s".format(current_time - start_time))
    #Get the preseverd filters after pruning by graph method based on pruning proportion
    block_index = 0
    stage = 0

    for name, module in origin_model.named_modules():
       
        if isinstance(module, BasicBlock):
            
            conv1_weight = module.conv1.weight.data
            _, _, centroids, indice = graph_weight(conv1_weight, int(conv1_weight.size(0) * (1 - pr_cfg[current_index])),logger)
            cfg.append(len(centroids))
            cfg.append(0) #assume baseblock has three conv layer
            centroids_state_dict[name + '.conv1.weight'] = centroids


        elif isinstance(module, Bottleneck):

            conv1_weight = module.conv1.weight.data
            _, _, centroids, indice = graph_weight(conv1_weight, int(conv1_weight.size(0) * (1 - pr_cfg[current_index])),logger)
            cfg.append(len(centroids))
            indices.append(indice)
            centroids_state_dict[name + '.conv1.weight'] = centroids.reshape((-1, conv1_weight.size(1), conv1_weight.size(2), conv1_weight.size(3)))
            
            current_index += 1

            conv2_weight = module.conv2.weight.data
            _, _, centroids, indice = graph_weight(conv2_weight, int(conv2_weight.size(0) * (1 - pr_cfg[current_index])),logger)
            cfg.append(len(centroids))
            centroids_state_dict[name + '.conv2.weight'] = centroids.reshape((-1, conv2_weight.size(1), conv2_weight.size(2), conv2_weight.size(3)))

            current_index+=1

    prune_state_dict.append('fc.weight')
    prune_state_dict.append('fc.bias')

    cfg.extend(block_cfg)
    #print(cfg)
    model = import_module(f'model.{args.arch}').resnet(args.cfg, layer_cfg=cfg).to(device)

    return model    

def test(model, testLoader, topk=(1,)):
    model.eval()

    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()

    start_time = time.time()
    with torch.no_grad():
        #i = 0
        for batch_idx, batch_data in enumerate(testLoader):
            #i += 1
            #if i > 2:
                #break
            inputs = batch_data[0]['data'].to(device)
            targets = batch_data[0]['label'].squeeze().long().to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets, topk=topk)
            accuracy.update(predicted[0], inputs.size(0))
            top5_accuracy.update(predicted[1], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                .format(float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), (current_time - start_time))
        )
    testLoader.reset()
    return accuracy.avg, top5_accuracy.avg
def main():

    print('==> Loading My Model..')
    resumeckpt = torch.load(args.resume)
    state_dict = resumeckpt['state_dict']
    cfg = resumeckpt['cfg']
        
    if args.arch == 'resnet_imagenet':
        if len(cfg) == 32:
            model = graph_resnet(args.pr_target).to(device)
        else:
            #print(1)
            model = import_module(f'model.{args.arch}').resnet(args.cfg, layer_cfg=cfg).to(device)
    elif args.arch == 'mobilenet_v1':
        model = import_module(f'model.{args.arch}').mobilenet_v1(layer_cfg=cfg).to(device)
    elif args.arch == 'mobilenet_v2':
        model = import_module(f'model.{args.arch}').mobilenet_v2(layer_cfg=cfg).to(device)
    else:
        raise('arch not exist!')

    model.load_state_dict(state_dict)

    #from torchsummaryX import summary
    #summary(model, torch.zeros((1, 3, 224, 224)).to(device))
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
    

    test(model, testLoader, topk=(1, 5))
    
if __name__ == '__main__':
    main()
