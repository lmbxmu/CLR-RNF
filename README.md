# GraphPruning



## GraphPruner

You can run the following code to prune model on CIFAR-10:
```shell
python cifar.py 
--arch vgg_cifar 
--cfg vgg16 
--data_path /data/cifar 
--job_dir ./experiment/cifar/vgg_1 
--pretrain_model /home/pretrain/vgg16_cifar10.pt 
--lr 0.01 
--lr_decay_step 50 100 
--weight_decay 0.005  
--num_epochs 150 
--gpus 0
--pr_target 0.7 
--graph_gpu
```


 You can run the following code to prune model on ImageNet: 

```shell
python imagenet.py 
--dataset imagenet 
--data_path /data/ImageNet/ 
--pretrain_model /data/model/resnet50.pth 
--job_dir /data/experiment/resnet50 
--arch resnet 
--cfg resnet50 
--lr 0.1 
--lr_decay_step 30 60 
--num_epochs 90 
--train_batch_size 256 
--weight_decay 1e-4 
--gpus 0 1 2 
--pr_target 0.7 
--graph_gpu
```
You can run the following code to get FLOPs prune ratio under a given parameters prune target:

```shell
python get_flops.py 
--arch resnet_imagenet 
--cfg resnet50 
--pretrain_model /media/disk2/zyc/prune_result/resnet_50/pruned_checkpoint/resnet50-19c8e357.pth 
--job_dir ./experiment/imagenet/resnet50_flop 
--graph_gpu 
--pr_target 0.1
```

You can run the following code to compare the loss between graph，Kmeans & random: 

```shell
python cal_graph_loss.py 
--arch vgg_cifar 
--cfg vgg16 
--data_path /data/cifar 
--job_dir ./experiment/vgg
--pretrain_model pretrain/vgg16_cifar10.pt 
--gpus 0 
--graph_gpu
```

## Pretrained model
| Model        | Download Link                                                |
| ------------ | ------------------------------------------------------------ |
| Mobilenet-v1 | https://hanlab.mit.edu/projects/amc/external/mobilenet_imagenet.pth.tar |
| Mobilenet-v2 | https://download.pytorch.org/models/mobilenet_v2-b0353104.pth |


## Other Arguments

```shell
optional arguments:
  -h, --help            show this help message and exit
  --gpus GPUS [GPUS ...]
                        Select gpu_id to use. default:[0]
  --dataset DATASET     Select dataset to train. default:cifar10
  --data_path DATA_PATH
                        The dictionary where the input is stored.
                        default:/home/data/cifar10/
  --job_dir JOB_DIR     The directory where the summaries will be stored.
                        default:./experiments
  --arch ARCH           Architecture of model. default:resnet
  --cfg CFG             Detail architecuture of model. default:resnet56
  --graph_gpu           Use gpu to graph the filters or not. default:False
  --init_method INIT_METHOD
                        Initital method of pruned model. default:direct.
                        optimal:random_project
  --pr_targt            Target prune ratio of parameters
```