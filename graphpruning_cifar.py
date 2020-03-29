import torch
import torch.nn as nn
import torch.optim as optim
from utils.options import args
from model.resnet_cifar import ResBasicBlock
from model.googlenet import Inception
import utils.common as utils

import os
import time
from data import cifar10
from importlib import import_module
from thop import profile

from utils.common import graph_weight, kmeans_weight, random_weight,random_project, direct_project

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()

# Data
print('==> Preparing data..')
loader = cifar10.Data(args)

# Load pretrained model
print('==> Loading pretrained model..')
if args.pretrain_model is None or not os.path.exists(args.pretrain_model):
    raise ('Pretrained_model path should be exist!')
ckpt = torch.load(args.pretrain_model, map_location=device)
if args.arch == 'vgg_cifar':
    origin_model = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)
elif args.arch == 'resnet_cifar':
    origin_model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
elif args.arch == 'googlenet':
    origin_model = import_module(f'model.{args.arch}').googlenet().to(device)
else:
    raise('arch not exist!')
origin_model.load_state_dict(ckpt['state_dict'])


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

    #load weight
    model = import_module(f'model.{args.arch}').VGG(args.cfg, layer_cfg=cfg).to(device)

    if args.init_method == 'random_project' or args.init_method == 'direct_project':
        pretrain_state_dict = origin_model.state_dict()
        state_dict = model.state_dict()
        centroids_state_dict_keys = list(centroids_state_dict.keys())


        for i, (k, v) in enumerate(centroids_state_dict.items()):
            if i == 0: #first conv need not to prune channel
                continue
            if args.init_method == 'random_project':
                centroids_state_dict[k] = random_project(torch.FloatTensor(centroids_state_dict[k]),
                                                         len(indices[i - 1]))
            else:
                centroids_state_dict[k] = direct_project(torch.FloatTensor(centroids_state_dict[k]), indices[i - 1])

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
    if args.init_method == 'random_project' or args.init_method == 'direct_project':
        pretrain_state_dict = origin_model.state_dict()
        state_dict = model.state_dict()
        centroids_state_dict_keys = list(centroids_state_dict.keys())
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

    if args.init_method == 'random_project' or args.init_method == 'direct_project':
        pretrain_state_dict = origin_model.state_dict()
        state_dict = model.state_dict()
        centroids_state_dict_keys = list(centroids_state_dict.keys())

        index = 0
        for k, v in centroids_state_dict.items():

            if k.endswith('.branch5x5.3.weight'):
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
       
def train(model, optimizer, trainLoader, args, epoch, topk=(1,)):

    model.train()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(trainLoader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets, topk=topk)
        accuracy.update(prec1[0], inputs.size(0))
        if len(topk) == 2:
            top5_accuracy.update(prec1[1], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            if len(topk) == 1:
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'Loss {:.4f}\t'
                    'Accuracy {:.2f}%\t\t'
                    'Time {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                        float(losses.avg), float(accuracy.avg), cost_time
                    )
                )
            else:
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'Loss {:.4f}\t'
                    'Top1 {:.2f}%\t'
                    'Top5 {:.2f}%\t'
                    'Time {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                        float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), cost_time
                    )
                )
            start_time = current_time

def test(model, testLoader, topk=(1,)):
    model.eval()

    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets, topk=topk)
            accuracy.update(predicted[0], inputs.size(0))
            if len(topk) == 2:
                top5_accuracy.update(predicted[1], inputs.size(0))

        current_time = time.time()
        if len(topk) == 1:
            logger.info(
                'Test Loss {:.4f}\tAccuracy {:.2f}%\t\tTime {:.2f}s\n'
                .format(float(losses.avg), float(accuracy.avg), (current_time - start_time))
            )
        else:
            logger.info(
                'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                    .format(float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), (current_time - start_time))
            )
    if len(topk) == 1:
        return accuracy.avg
    else:
        return top5_accuracy.avg

def main():
    start_epoch = 0
    best_acc = 0.0
    test(origin_model,loader.testLoader, topk=(1, 5) if args.dataset == 'imagenet' else (1, ))
    print('==> Building Model..')
    if args.pretrain_model is None or not os.path.exists(args.pretrain_model):
        raise ('Pretrained_model path should be exist!')
    if args.arch == 'vgg_cifar':
        model, cfg = graph_vgg(args.pr_target)
    elif args.arch == 'resnet_cifar':
        model, cfg = graph_resnet(args.pr_target)
    elif args.arch == 'googlenet':
        model, cfg = graph_googlenet(args.pr_target)
    else:
        raise('arch not exist!')
    print("Graph Down!")

    #Calculate the flops and params of origin_model & pruned_model
    Input = torch.randn(1, 3, 32, 32).to(device)
    oriflops, oriparams = profile(origin_model, inputs=(Input, ))
    flops, params = profile(model, inputs=(Input, ))

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)


    for epoch in range(start_epoch, args.num_epochs):
        train(model, optimizer, loader.trainLoader, args, epoch, topk=(1, 5) if args.dataset == 'imagenet' else (1, ))
        scheduler.step()
        test_acc = test(model, loader.testLoader, topk=(1, 5) if args.dataset == 'imagenet' else (1, ))

        is_best = best_acc < test_acc
        best_acc = max(best_acc, test_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'arch': args.cfg,
            'cfg': cfg
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best accuracy: {:.3f}'.format(float(best_acc)))


    orichannel = 0
    channel = 0

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