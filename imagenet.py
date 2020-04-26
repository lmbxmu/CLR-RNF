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


device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
if args.criterion == 'Softmax':
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
elif args.criterion == 'SmoothSoftmax':
    criterion = CrossEntropyLabelSmooth(1000,args.label_smooth)
    criterion = criterion.cuda()
else:
    raise ValueError('invalid criterion : {:}'.format(args.criterion))
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

flops_cfg = {
    'resnet50':[2]*3+[1.5]*4+[1]*6+[0.5]*3,
    #'resnet50':[1.0, 1.67262, 0.3869, 1.26786, 0.3869, 1.26786, 0.77381, 2.02679, 0.38393, 1.25298, 0.38393, 1.25298, 0.38393, 1.25298, 0.76786, 2.01339, 0.38244, 1.24554, 0.38244, 1.24554, 0.38244, 1.24554, 0.38244, 1.24554, 0.38244, 1.24554, 0.76488, 2.0067, 0.3817, 1.24182, 0.3817, 1.24182]
}
flops_lambda = {
 'resnet50':1,
}

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
    origin_model.load_state_dict(ckpt)
else:
    raise('arch not exist!')


def graph_resnet(pr_target):

    pr_cfg = []
    weights = []

    cfg = []
    centroids_state_dict = {}
    prune_state_dict = []
    indices = []

    current_index = 0
    index = 0
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
    if args.init_method == 'random_project' or args.init_method == 'direct_project':
        pretrain_state_dict = origin_model.state_dict()
        state_dict = model.state_dict()
        centroids_state_dict_keys = list(centroids_state_dict.keys())

        index = 0
        for k, v in centroids_state_dict.items():

            if k.endswith('.conv2.weight') and args.cfg != 'resnet18' and args.cfg != 'resnet34':
                if args.init_method == 'random_project':
                    centroids_state_dict[k] = random_project(torch.FloatTensor(centroids_state_dict[k]),
                                                             len(indices[index]))
                else:
                    centroids_state_dict[k] = direct_project(torch.FloatTensor(centroids_state_dict[k]), indices[index])
                index += 1

        for k, v in state_dict.items():
            if k in prune_state_dict:
                continue
            elif k in centroids_state_dict_keys:
                state_dict[k] = torch.FloatTensor(centroids_state_dict[k]).view_as(state_dict[k])
            else:
                state_dict[k] = pretrain_state_dict[k]
        model.load_state_dict(state_dict)
    else:
        pass
    return model, cfg    

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
    '''
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())
    for param_tensor in centroids_state_dict:
        print(param_tensor, '\t', centroids_state_dict[param_tensor].size())
    '''
    if args.init_method == 'random_project' or args.init_method == 'direct_project':
        pretrain_state_dict = origin_model.state_dict()
        state_dict = model.state_dict()
        centroids_state_dict_keys = list(centroids_state_dict.keys())

        for i, (k, v) in enumerate(centroids_state_dict.items()):
            if i == 0 or i % 2 == 1: #first conv and dw conv need not to prune channel
                continue
            if args.init_method == 'random_project':
                centroids_state_dict[k] = random_project(torch.FloatTensor(centroids_state_dict[k]),
                                                         len(indices[i - 2]))
            else:
                centroids_state_dict[k] = direct_project(torch.FloatTensor(centroids_state_dict[k]), indices[i - 2])

        for k, v in state_dict.items():
            if k in prune_state_dict:
                continue
            elif k in centroids_state_dict_keys:
                state_dict[k] = torch.FloatTensor(centroids_state_dict[k]).view_as(state_dict[k])
            else:
                state_dict[k] = pretrain_state_dict[k]
        model.load_state_dict(state_dict)
    else:
        pass
    return model, cfg   

def graph_mobilenet_v2(pr_target):
    pr_cfg = []
    weights = []

    cfg = []
    centroids_state_dict = {}
    prune_state_dict = []
    current_index = 0
    cat_cfg = [2,3,4,3,3,1]
    c_index, i_index = 0, 0 
    # Sort the weights and get the pruning threshold
    for name, module in origin_model.named_modules():

        if isinstance(module, InvertedResidual):
            conv1_weight = module.conv[3].weight.data
            if len(module.conv) == 5: #expand_ratio = 1
                weights.append(conv1_weight.view(-1))
            else:   
                conv2_weight = module.conv[6].weight.data       
                if i_index == 0:
                    conv_weight = torch.cat((conv1_weight.view(-1),conv2_weight.view(-1)),0)
                else:
                    conv_weight = torch.cat((conv_weight,torch.cat((conv1_weight.view(-1),conv2_weight.view(-1)),0)),0)
                i_index += 1
                if i_index == cat_cfg[c_index]:
                    weights.append(conv_weight)  
                    c_index += 1
                    i_index = 0
                

    #weights.append(origin_model.state_dict()['features.18.0.weight'].view(-1))  #lastlayer          

    all_weights = torch.cat(weights, 0)
    preserve_num = int(all_weights.size(0) * (1 - pr_target))
    preserve_weight, _ = torch.topk(torch.abs(all_weights), preserve_num)
    threshold = preserve_weight[preserve_num - 1]

    # Based on the pruning threshold, the prune cfg of each layer is obtained
    for weight in weights:
        pr_cfg.append(torch.sum(torch.lt(torch.abs(weight), threshold)).item() / weight.size(0))
    #print(pr_cfg)
    graph_cfg = []
    graph_cfg.append(pr_cfg[0])
    for i in range(6):
        for j in range(cat_cfg[i]):
            graph_cfg.append(pr_cfg[i+1])
    #print(graph_cfg)

    # Get the preseverd filters after pruning by graph method based on pruning proportion
    for name, module in origin_model.named_modules():

        if isinstance(module, InvertedResidual):
    
            if len(module.conv) == 5: #expand_ratio = 1 first layer

                conv1_weight = module.conv[3].weight.data
                _, _, centroids, indice = graph_weight(conv1_weight, int(conv1_weight.size(0) * (1 - graph_cfg[current_index])),logger)
                centroids_state_dict[name + '.conv.3.weight'] = centroids
                lastindice = indice

                prune_state_dict.append(name + '.conv.4.weight')
                prune_state_dict.append(name + '.conv.4.bias')
                prune_state_dict.append(name + '.conv.4.running_var')
                prune_state_dict.append(name + '.conv.4.running_mean')
                current_index += 1

            else:

                conv1_weight = module.conv[0].weight.data
                _, _, centroids, indice1 = graph_weight(conv1_weight,
                                                       int(conv1_weight.size(0) * (1 - graph_cfg[current_index-1])), logger)

                centroids_state_dict[name + '.conv.0.weight'] = centroids.reshape((-1, conv1_weight.size(1), conv1_weight.size(2), conv1_weight.size(3)))
                prune_state_dict.append(name + '.conv.1.weight')
                prune_state_dict.append(name + '.conv.1.bias')
                prune_state_dict.append(name + '.conv.1.running_var')
                prune_state_dict.append(name + '.conv.1.running_mean')

                conv2_weight = module.conv[3].weight.data
                _, _, centroids, indice2 = graph_weight(conv2_weight,
                                                       int(conv2_weight.size(0) * (1 - graph_cfg[current_index-1])), logger)
                centroids_state_dict[name + '.conv.3.weight'] = centroids
                prune_state_dict.append(name + '.conv.4.weight')
                prune_state_dict.append(name + '.conv.4.bias')
                prune_state_dict.append(name + '.conv.4.running_mean')
                prune_state_dict.append(name + '.conv.4.running_var')

                conv3_weight = module.conv[6].weight.data
                _, _, centroids, indice3 = graph_weight(conv3_weight,
                                                       int(conv3_weight.size(0) * (1 - graph_cfg[current_index])), logger)
                centroids_state_dict[name + '.conv.6.weight'] = centroids.reshape((-1, conv3_weight.size(1), conv3_weight.size(2), conv3_weight.size(3)))
                prune_state_dict.append(name + '.conv.7.weight')
                prune_state_dict.append(name + '.conv.7.bias')
                prune_state_dict.append(name + '.conv.7.running_mean')
                prune_state_dict.append(name + '.conv.7.running_var')

                if args.init_method == 'random_project':
                    centroids_state_dict[name + '.conv.0.weight'] = random_project(centroids_state_dict[name + '.conv.0.weight'], len(lastindice))
                    centroids_state_dict[name + '.conv.6.weight'] = random_project(centroids_state_dict[name + '.conv.6.weight'], len(indice2))
                else:
                    centroids_state_dict[name + '.conv.0.weight'] = direct_project(centroids_state_dict[name + '.conv.0.weight'], lastindice)
                    centroids_state_dict[name + '.conv.6.weight'] = direct_project(centroids_state_dict[name + '.conv.6.weight'], indice2)

                lastindice = indice3
                current_index += 1
    '''
    #LastLayer
    conv_weight = origin_model.state_dict()['features.18.0.weight']
    _, _, centroids, indice = graph_weight(conv_weight, int(conv_weight.size(0) * (1 - pr_cfg[current_index])),logger)
    centroids_state_dict['features.18.0.weight'] = centroids.reshape((-1, conv_weight.size(1), conv_weight.size(2), conv_weight.size(3)))

    prune_state_dict.append('features.18.1.weight')
    prune_state_dict.append('features.18.1.bias')
    prune_state_dict.append('features.18.1.running_var')
    prune_state_dict.append('features.18.1.running_mean')
    prune_state_dict.append('classifier.1.weight')
    prune_state_dict.append('classifier.1.bias')
    '''

    if args.init_method == 'random_project':
        centroids_state_dict['features.18.0.weight'] = random_project(origin_model.state_dict()['features.18.0.weight'], len(lastindice))
    else:
        centroids_state_dict['features.18.0.weight'] = direct_project(origin_model.state_dict()['features.18.0.weight'], lastindice)
    
    model = import_module(f'model.{args.arch}').mobilenet_v2(layer_cfg=graph_cfg).to(device)

    if args.init_method == 'random_project' or args.init_method == 'direct_project':
        pretrain_state_dict = origin_model.state_dict()
        state_dict = model.state_dict()
        centroids_state_dict_keys = list(centroids_state_dict.keys())

        #for param_tensor in state_dict:
            #print(param_tensor,'\t',state_dict[param_tensor].size())

        #for param_tensor in centroids_state_dict:
            #print(param_tensor,'\t',centroids_state_dict[param_tensor].size())

        for k, v in state_dict.items():
            if k in prune_state_dict:
                continue
            elif k in centroids_state_dict_keys:
                state_dict[k] = torch.FloatTensor(centroids_state_dict[k]).view_as(state_dict[k])
            else:
                state_dict[k] = pretrain_state_dict[k]
        model.load_state_dict(state_dict)
    else:
        pass
    return model, cfg

def train(model, optimizer, trainLoader, args, epoch, topk=(1,)):

    model.train()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()
    print_freq = trainLoader._size // args.train_batch_size // 10
    start_time = time.time()
    #i = 0
    for batch, batch_data in enumerate(trainLoader):

        inputs = batch_data[0]['data'].to(device)
        targets = batch_data[0]['label'].squeeze().long().to(device)
        #i += 1
        #if i > 2:
            #break
        adjust_learning_rate(optimizer, epoch, batch, trainLoader._size // args.train_batch_size)

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets, topk=topk)
        accuracy.update(prec1[0], inputs.size(0))
        top5_accuracy.update(prec1[1], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Top1 {:.2f}%\t'
                'Top5 {:.2f}%\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.train_batch_size, trainLoader._size,
                    float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), cost_time
                )
            )
            start_time = current_time
    trainLoader.reset()

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

def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    #Warmup
    if args.lr_type == 'step':
        factor = epoch // 30

        if epoch >= 80:
            factor = factor + 1

        lr = args.lr * (0.1 ** factor)

    elif args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.num_epochs))
    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))
    elif args.lr_type == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError
    if epoch < 5:
            lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)
    print('=> lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    start_epoch = 0
    best_top1_acc = 0.0
    best_top5_acc = 0.0

    test(origin_model, testLoader, topk=(1, 5))
    print('==> Building Model..')
    if args.pretrain_model is None or not os.path.exists(args.pretrain_model):
        raise ('Pretrained_model path should be exist!')
    if args.arch == 'resnet_imagenet':
        model, cfg = graph_resnet(args.pr_target)
    elif args.arch == 'mobilenet_v1':
        model, cfg = graph_mobilenet_v1(args.pr_target)
    elif args.arch == 'mobilenet_v2':
        model, cfg = graph_mobilenet_v2(args.pr_target)
    else:
        raise('arch not exist!')
    print("Graph Down!")

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(start_epoch, args.num_epochs):
        train(model, optimizer, trainLoader, args, epoch, topk=(1, 5))

        test_top1_acc, test_top5_acc = test(model, testLoader, topk=(1, 5))

        is_best = best_top5_acc < test_top5_acc
        best_top1_acc = max(best_top1_acc, test_top1_acc)
        best_top5_acc = max(best_top5_acc, test_top5_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_top1_acc': best_top1_acc,
            'best_top5_acc': best_top5_acc,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'arch': args.cfg,
            'cfg': cfg
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best Top-1 accuracy: {:.3f} Top-5 accuracy: {:.3f}'.format(float(best_top1_acc), float(best_top5_acc)))

    
if __name__ == '__main__':
    main()
