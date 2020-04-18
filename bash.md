python get_flops.py --arch vgg_cifar --cfg vgg16 --job_dir ./experiment/vgg_cifar --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/vgg16_cifar10.pt --pr_target 0.8

python cal_graph_loss.py --arch vgg_cifar --cfg vgg16 --job_dir ./experiment/vgg_cifar --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/vgg16_cifar10.pt

python cifar.py --arch vgg_cifar --cfg vgg16 --job_dir ./experiment/vgg_cifar --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/vgg16_cifar10.pt --pr_target 0.8

python cifar.py --arch resnet_cifar --cfg resnet56 --job_dir ./experiment/resnet_cifar --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/resnet_56.pt --pr_target 0.8

python cifar.py --arch googlenet --cfg googlenet --job_dir ./experiment/googlenet --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/googlenet.pt --pr_target 0.8

python graphpruning_cifar.py --arch vgg_cifar --cfg vgg16 --data_path /home/lmb/ABCPrunerPlus/data/cifar --job_dir ./experiment/vgg_cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/vgg16_cifar10.pt --gpus 0 1 --graph_gpu

python graphpruning_cifar.py --arch resnet_cifar --cfg resnet56 --job_dir ./experiment/resnet56_cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_56.pt --gpus 0 --graph_gpu

python graphpruning_cifar.py --arch resnet_cifar --cfg resnet110 --job_dir ./experiment/resnet110_cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_110.pt --gpus 0 --graph_gpu

python graphpruning_cifar.py --arch googlenet --cfg googlenet --job_dir ./experiment/googlenet --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/googlenet.pt --gpus 0 --graph_gpu 


python graphpruning_imagenet.py --arch resnet_imagenet --cfg resnet18 --job_dir ./experiment/resnet18 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet18.pth --gpus 0 --graph_gpu 

python graphpruning_imagenet.py --arch resnet --cfg resnet34 --job_dir ./experiment/resnet34 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet34.pth --gpus 0 --graph_gpu 

python graphpruning_imagenet.py --arch resnet_imagenet --cfg resnet50 --job_dir ./experiment/resnet50 --pretrain_model /media/disk2/zyc/prune_result/resnet_50/pruned_checkpoint/resnet50-19c8e357.pth --gpus 0 --graph_gpu 

python graphpruning_imagenet.py --arch resnet --cfg resnet101 --job_dir ./experiment/resnet101 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet101.pth --gpus 0 --graph_gpu 

python graphpruning_imagenet.py --arch resnet --cfg resnet152 --job_dir ./experiment/resnet152 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet152.pth --gpus 0 --graph_gpu 


python cal_graph_loss.py --arch vgg_cifar --cfg vgg16 --data_path /home/lmb/ABCPrunerPlus/data/cifar --job_dir ./experiment/vgg_cifar_4 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/vgg16_cifar10.pt --gpus 0 --graph_gpu

python cal_graph_loss.py --arch vgg_cifar --cfg vgg16 --data_path /home/lmb/ABCPrunerPlus/data/cifar --job_dir ./experiment/vgg_cifar_5 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/vgg16_cifar10.pt --gpus 0 --graph_gpu

python cal_graph_loss.py --arch vgg_cifar --cfg vgg16 --data_path /home/lmb/ABCPrunerPlus/data/cifar --job_dir ./experiment/vgg_cifar_6 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/vgg16_cifar10.pt --gpus 0 --graph_gpu

#cifar experiment
python graphpruning_cifar.py --arch vgg_cifar --cfg vgg16 --data_path /home/lmb/ABCPrunerPlus/data/cifar --job_dir ./experiment/cifar/vgg_1 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/vgg16_cifar10.pt --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 1 --pr_target 0.7 --graph_gpu
python graphpruning_cifar.py --arch vgg_cifar --cfg vgg16 --data_path /home/lmb/ABCPrunerPlus/data/cifar --job_dir ./experiment/cifar/vgg_2 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/vgg16_cifar10.pt --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 2 3 --pr_target 0.8 --graph_gpu
python graphpruning_cifar.py --arch vgg_cifar --cfg vgg16 --data_path /home/lmb/ABCPrunerPlus/data/cifar --job_dir ./experiment/cifar/vgg_3 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/vgg16_cifar10.pt --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 1 --pr_target 0.85 --graph_gpu
python graphpruning_cifar.py --arch vgg_cifar --cfg vgg16 --data_path /home/lmb/ABCPrunerPlus/data/cifar --job_dir ./experiment/cifar/vgg_4 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/vgg16_cifar10.pt --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 2 3 --pr_target 0.85 --graph_gpu
python graphpruning_cifar.py --arch vgg_cifar --cfg vgg16 --data_path /home/lmb/ABCPrunerPlus/data/cifar --job_dir ./experiment/cifar/vgg_5 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/vgg16_cifar10.pt --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 --pr_target 0.9 --graph_gpu
python graphpruning_cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_56.pt  --job_dir ./experiment/cifar/res56_3 --arch resnet_cifar --cfg resnet56 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 1 --pr_target 0.6 --graph_gpu
python graphpruning_cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_56.pt  --job_dir ./experiment/cifar/res56_1 --arch resnet_cifar --cfg resnet56 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 1 --pr_target 0.7 --graph_gpu
python graphpruning_cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_56.pt  --job_dir ./experiment/cifar/res56_2 --arch resnet_cifar --cfg resnet56 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 2 3 --pr_target 0.8 --graph_gpu
python graphpruning_cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_110.pt  --job_dir ./experiment/cifar/res110_1 --arch resnet_cifar --cfg resnet110 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 1 --pr_target 0.7
python graphpruning_cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_110.pt  --job_dir ./experiment/cifar/res110_3 --arch resnet_cifar --cfg resnet110 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 2 3 --pr_target 0.65
python graphpruning_cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_110.pt  --job_dir ./experiment/cifar/res110_2 --arch resnet_cifar --cfg resnet110 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 2 3 --pr_target 0.8
python graphpruning_cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/googlenet.pt  --job_dir ./experiment/cifar/googlenet_1 --arch googlenet --cfg googlenet --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 1 2 3 --pr_target 0.7 --graph_gpu

python graphpruning_cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_56.pt  --job_dir ./experiment/cifar/res56_4 --arch resnet_cifar --cfg resnet56 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 2 3 --pr_target 0.55 --graph_gpu
python graphpruning_cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_110.pt  --job_dir ./experiment/cifar/res110_4 --arch resnet_cifar --cfg resnet110 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 1 --pr_target 0.75

python graphpruning_cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_56.pt  --job_dir ./experiment/cifar/res56_5 --arch resnet_cifar --cfg resnet56 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 2 --pr_target 0.55 --graph_gpu --init_method random_project

python graphpruning_cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_56.pt  --job_dir ./experiment/cifar/res56_8 --arch resnet_cifar --cfg resnet56 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 --pr_target 0.54 --graph_gpu

python graphpruning_cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_110.pt  --job_dir ./experiment/cifar/res110_5 --arch resnet_cifar --cfg resnet110 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 1 --pr_target 0.75 --init_method random_project

python graphpruning_cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_110.pt  --job_dir ./experiment/cifar/res110_5_1 --arch resnet_cifar --cfg resnet110 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 2 3 --pr_target 0.75 --init_method random_project



python cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/googlenet.pt  --job_dir ./experiment/cifar/googlenet_3 --arch googlenet --cfg googlenet --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 1 2 3 --pr_target 0.95 --graph_gpu

python cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_56.pt  --job_dir ./experiment/cifar/res56_9 --arch resnet_cifar --cfg resnet56 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 --pr_target 0.53 --graph_gpu
python cifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_56.pt  --job_dir ./experiment/cifar/res56_10 --arch resnet_cifar --cfg resnet56 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 --pr_target 0.52 --graph_gpu
python ifar.py --data_path /home/lmb/ABCPrunerPlus/data/cifar --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet_56.pt  --job_dir ./experiment/cifar/res56_11 --arch resnet_cifar --cfg resnet56 --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005  --num_epochs 150 --gpus 0 --pr_target 0.51 --graph_gpu

#imagenet experiment

python graphpruning_imagenet.py --arch resnet_imagenet --cfg resnet18 --data_path /media/disk2/zyc/ImageNet2012 --pretrain_model /home/lmb/ABCPrunerPlus/pretrain/resnet18.pth --job_dir ./experiment/imagenet/resnet18 --lr 0.1 --weight_decay 0.0001 --num_epochs 90 --lr_decay_step 30 --gpus 0 1 --train_batch_size 256 --eval_batch_size 256 --pr_target 0.8 

python imagenet.py --arch resnet_imagenet --cfg resnet50 --data_path /media/disk2/zyc/ImageNet2012 --pretrain_model /media/disk2/zyc/prune_result/resnet_50/pruned_checkpoint/resnet50-19c8e357.pth --job_dir ./experiment/imagenet/resnet50 --lr 0.1 --weight_decay 0.0001 --num_epochs 90 --lr_decay_step 30 --gpus 0 1 --train_batch_size 256 --eval_batch_size 256 --pr_target 0.8 

python get_flops.py --arch resnet_imagenet --cfg resnet50 --pretrain_model /media/disk2/zyc/prune_result/resnet_50/pruned_checkpoint/resnet50-19c8e357.pth --job_dir ./experiment/imagenet/resnet50_flop --graph_gpu --pr_target 0.1

python imagenet.py --arch mobilenet_v2 --cfg mobilenet_v2 --data_path /media/disk2/zyc/ImageNet2012 --pretrain_model ./pretrain/checkpoints/mobilenet_v2.pth.tar --job_dir ./experiment/imagenet/mobilenet_v2 --lr 0.1 --weight_decay 4e-5 --num_epochs 150 --gpus 0 1 --train_batch_size 256 --eval_batch_size 256 --pr_target 0.25

python imagenet.py --arch mobilenet_v2 --cfg mobilenet_v2 --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/mobilenet_v2.pth.tar --job_dir ./experiment/imagenet/mobilenet_v2 --pr_target 0.25

python get_flops.py --arch mobilenet_v2 --cfg mobilenet_v2 --pretrain_model /Users/zhangyuxin/Documents/MAC/pretrain_model/mobilenet_v2.pth.tar --job_dir ./experiment/imagenet/mobilenet_v2 --pr_target 0.5

python get_flops.py --arch mobilenet_v2 --cfg mobilenet_v2 --pretrain_model ./pretrain/checkpoints/mobilenet_v2.pth.tar --job_dir ./experiment/imagenet/mobilenet_v2 --pr_target 0.5

python graphpruning_imagenet.py --arch mobilenet_v1 --cfg mobilenet_v1 --data_path /media/disk2/zyc/ImageNet2012 --pretrain_model ./pretrain/checkpoints/mobilenet_imagenet.pth.tar --job_dir ./experiment/imagenet/mobilenetv1 --lr 0.1 --weight_decay 0.0001 --num_epochs 90 --lr_decay_step 30 --gpus 0 1 --train_batch_size 256 --eval_batch_size 256 --pr_target 0.8 