cd GraphPruning
conda activate pt1.0
#debug
##vgg16
python get_flops.py --arch vgg_cifar --cfg vgg16 --job_dir ./experiment/vgg_cifar --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/vgg16_cifar10.pt --pr_target 0.8
python cifar.py --arch vgg_cifar --cfg vgg16 --job_dir ./experiment/vgg_cifar --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/vgg16_cifar10.pt --pr_target 0.8 --graph_method kmeans --init_method random_project
python cal_graph_loss.py --arch vgg_cifar --cfg vgg16 --job_dir ./experiment/vgg_cifar --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/vgg16_cifar10.pt

##resnet56
python get_flops.py --arch resnet_cifar --cfg resnet56 --job_dir ./experiment/resnet_cifar --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/resnet_56.pt --pr_target 0.6
python cifar.py --arch resnet_cifar --cfg resnet56 --job_dir ./experiment/resnet_cifar --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/resnet_56.pt --pr_target 0.6

##resnet110
python get_flops.py --arch resnet_cifar --cfg resnet110 --job_dir ./experiment/resnet_cifar --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/resnet_110.pt --pr_target 0.57
python cifar.py --arch resnet_cifar --cfg resnet110 --job_dir ./experiment/resnet_cifar --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/resnet_110.pt --pr_target 0.57

##googlenet
python get_flops.py --arch googlenet --cfg googlenet --job_dir ./experiment/googlenet --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/googlenet.pt --pr_target 0.8
python cifar.py --arch googlenet --cfg googlenet --job_dir ./experiment/googlenet --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/googlenet.pt --pr_target 0.8

##resnet50

python get_flops.py --arch resnet_imagenet --cfg resnet50 --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/resnet50.pth --job_dir ./experiment/imagenet/resnet50_flop --graph_gpu --pr_target 0.7
python imagenet.py --arch resnet_imagenet --cfg resnet50 --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/resnet50.pth --job_dir ./experiment/imagenet/resnet50 --pr_target 0.25

##mobilenet
python imagenet.py --arch mobilenet_v2 --cfg mobilenet_v2 --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/mobilenetv2_1.0-f2a8633.pth.tar --job_dir ./experiment/imagenet/mobilenet_v2 --pr_target 0.5

python get_flops.py --arch mobilenet_v2 --cfg mobilenet_v2 --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/mobilenetv2_1.0-f2a8633.pth.tar --job_dir ./experiment/imagenet/mobilenet_v2 --pr_target 0.5 --dataset imagenet

python cifar.py --arch mobilenetv2_cifar --cfg mobilenet_v2 --data_path /home/lmb/ABCPrunerPlus/data/cifar --job_dir ./experiment/cifar/mobilenetv2_baseline --lr 0.1 --lr_type cos --weight_decay 5e-4  --num_epochs 300 --gpus 0 

python cifar.py --arch mobilenetv2_cifar --cfg mobilenet_v2 --data_path /home/lmb/ABCPrunerPlus/data/cifar --job_dir ./experiment/cifar/mobilenetv2_baseline_2 --lr 0.1 --lr_decay_step 150 225 --weight_decay 5e-4 --num_epochs 300 --gpus 0 

#cifar experiment
##vgg
python get_flops.py --arch vgg_cifar --cfg vgg16 --job_dir ./experiment/flop/cgg --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/vgg16_cifar10.pt --pr_target 0.56
python cifar.py --arch vgg_cifar --cfg vgg16 --data_path /home/lmb/ABCPrunerPlus/data/cifar --job_dir ./experiment/cifar/vgg_step4_1 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/vgg16_cifar10.pt --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 --pr_target 0.86 --graph_method knn
##resnet56
python get_flops.py --arch resnet_cifar --cfg resnet56 --pretrain_model /home/lmb/CLR-RNF-master/pretrain/resnet_56.pt --job_dir ./experiment/flop/res56 --pr_target 0.25
python cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/CLR-RNF-master/pretrain/resnet_56.pt  --job_dir ./experiment/cifar/res56_1 --arch resnet_cifar --cfg resnet56 --lr 0.01 --lr_decay_step 150 225 --weight_decay 0.005 --dataset cifar10 --num_epochs 150 --gpus 0 --pr_target 0.25
##resnet110
python get_flops.py --arch resnet_cifar --cfg resnet110 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_110.pt --job_dir ./experiment/flop/res110--pr_target 0.7
python cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_110.pt  --job_dir ./experiment/cifar/res110_step_4_1 --arch resnet_cifar --cfg resnet110 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 1 --pr_target 0.69

##googlenet
python cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/googlenet.pt  --job_dir ./experiment/cifar/googlenet_times --arch googlenet --cfg googlenet --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 1 --pr_target 0.95
python get_flops.py --arch googlenet --cfg googlenet --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/googlenet.pt --job_dir ./experiment/flop/googlenet --pr_target 0.92

python get_flops.py --arch resnet_imagenet --cfg resnet50 --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/resnet50.pth --pr_target 0.38 --dataset imagenet

##mobilenet
python cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_110.pt  --job_dir ./experiment/cifar/mobilenetv2_1 --arch mobilenetv2_cifar --cfg mobilenet_v2--lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 1 --pr_target 0.69

#ablation study

##vgg
python cifar.py --arch vgg_cifar --cfg vgg16 --data_path /home/lmb/ABCPrunerPlus/data/cifar --job_dir ./experiment/cifar/vgg_kmeans_2 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/vgg16_cifar10.pt --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 --pr_target 0.86 --graph_method kmeans --init_method random_project
python cifar.py --arch vgg_cifar --cfg vgg16 --data_path /home/lmb/ABCPrunerPlus/data/cifar --job_dir ./experiment/cifar/vgg_abc --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/vgg16_cifar10.pt --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 --pr_target 0.86 
python get_flops.py --arch vgg_cifar --cfg vgg16 --job_dir ./experiment/flop/vgg --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/vgg16_cifar10.pt --graph_method kmeans --init_method random_project --pr_target 0.56 

##resnet56
python get_flops.py --arch resnet_cifar --cfg resnet56 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_56.pt --job_dir ./experiment/flop/res56 --pr_target 0.56 --graph_method kmeans --init_method random_project
python cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_56.pt  --job_dir ./experiment/cifar/res56_kmeans_2 --arch resnet_cifar --cfg resnet56 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 3 --pr_target 0.56 --graph_method kmeans --init_method random_project 

##resnet110
python get_flops.py --arch resnet_cifar --cfg resnet110 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_110.pt --job_dir ./experiment/flop/res110--pr_target 0.7 --graph_method kmeans --init_method random_project
python cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_110.pt  --job_dir ./experiment/cifar/res110_kmeans_2 --arch resnet_cifar --cfg resnet110 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 2 --pr_target 0.69 --graph_method kmeans --init_method random_project 
##googlenet
python cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/googlenet.pt  --job_dir ./experiment/cifar/googlenet_kmeans --arch googlenet --cfg googlenet --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 2 3 --pr_target 0.91 --graph_method kmeans --init_method random_project
python get_flops.py --arch googlenet --cfg googlenet --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/googlenet.pt --job_dir ./experiment/cifar/googlenet_kmeans 1 --pr_target 0.91 --graph_method kmeans --init_method random_project --gpus 0 2 3


#ablation study 2
python cifar.py --arch vgg_cifar --cfg vgg16 --data_path /home/lmb/ABCPrunerPlus/data/cifar --job_dir ./experiment/cifar/vgg_human --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/vgg16_cifar10.pt --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 --pr_target 0.86
python cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_56.pt  --job_dir ./experiment/cifar/res56_human --arch resnet_cifar --cfg resnet56 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --pr_target 0.56 --gpus 2 
python cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_110.pt  --job_dir ./experiment/cifar/res110_human --arch resnet_cifar --cfg resnet110 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 1 3 --pr_target 0.69
python cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/googlenet.pt  --job_dir ./experiment/cifar/googlenet_human --arch googlenet --cfg googlenet --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 1 2 3 --pr_target 0.91


#test
python test.py --arch resnet_imagenet --cfg resnet50 --data_path /media/disk2/zyc/ImageNet2012 --resume ./pretrain/checkpoints/model_best.pt --pretrain_model /media/disk2/zyc/prune_result/resnet_50/pruned_checkpoint/resnet50-19c8e357.pth --pr_target 0.2 --job_dir ./experiment/imagenet/test

#imagenet experiment

##resnet18
python imagenet.py --arch resnet_imagenet --cfg resnet18 --data_path /media/disk2/zyc/ImageNet2012 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet18.pth --job_dir ./experiment/imagenet/resnet18 --lr 0.1 --weight_decay 0.0001 --num_epochs 90 --lr_decay_step 30 --gpus 0 1 --train_batch_size 256 --eval_batch_size 256 --pr_target 0.8 

##resnet50
python imagenet_past.py --arch resnet_imagenet --cfg resnet50 --data_path /media/disk2/zyc/ImageNet2012 --pretrain_model /media/disk2/zyc/prune_result/resnet_50/pruned_checkpoint/resnet50-19c8e357.pth --job_dir ./experiment/imagenet/resnet50_redidual --lr 0.1 --weight_decay 0.0001 --num_epochs 90 --gpus 0 --train_batch_size 8 --eval_batch_size 8 --pr_target 0.48

python imagenet.py --arch resnet_imagenet --cfg resnet50 --data_path /media/disk2/zyc/ImageNet2012 --pretrain_model /media/disk2/zyc/prune_result/resnet_50/pruned_checkpoint/resnet50-19c8e357.pth --job_dir ./experiment/imagenet/resnet50_redidual --lr 0.1 --weight_decay 0.0001 --num_epochs 90 --gpus 0 --train_batch_size 8 --eval_batch_size 8 --resume ./experiment/imagenet/resnet50_redidual/checkpoint/model_last.pt

python get_flops.py --arch resnet_imagenet --cfg resnet50 --pretrain_model /media/disk2/zyc/prune_result/resnet_50/pruned_checkpoint/resnet50-19c8e357.pth --job_dir ./experiment/imagenet/resnet50_flop --graph_gpu --dataset imagenet --pr_target 0.48

##mobilenetv2
python imagenet.py --arch mobilenet_v2 --cfg mobilenet_v2 --data_path /media/disk2/zyc/ImageNet2012 --pretrain_model ./pretrain/checkpoints/mobilenet_v2_past.pth.tar --lr 0.1 --weight_decay 4e-5 --num_epochs 150 --gpus 1 --train_batch_size 8 --eval_batch_size 8 --pr_target 0.55 --lr_type cos --job_dir ./experiment/imagenet/mobilenet_v2_test

python get_flops.py --arch mobilenet_v2 --cfg mobilenet_v2 --pretrain_model ./pretrain/checkpoints/mobilenet_v2_past.pth.tar --job_dir ./experiment/imagenet/mobilenet_v2 --pr_target 0.15 --dataset imagenet


##mobilenetv1
python imagenet.py --arch mobilenet_v1 --cfg mobilenet_v1 --data_path /media/disk2/zyc/ImageNet2012 --pretrain_model ./pretrain/checkpoints/mobilenet_imagenet.pth.tar --job_dir ./experiment/imagenet/mobilenetv1 --lr 0.1 --weight_decay 0.0001  --gpus 0 1 --train_batch_size 8 --eval_batch_size 8 --pr_target 0.8 --num_epochs 150 --lr_type cos --use_dali