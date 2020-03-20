# GraphPruning

Run the following code:

```shell
python graphpruning_cifar.py
--arch vgg_cifar 
--cfg vgg16 
--job_dir ./experiment/vgg_cifar 
--pretrain_model /data/model/vgg16.pt 
--graph_gpu
```

## Other Arguments

```shell
optional arguments:
  -h, --help            show this help message and exit
  --gpus GPUS [GPUS ...]
                        Select gpu_id to use. default:[0]
  --dataset DATASET     Select dataset to train. default:cifar10
  --data_path DATA_PATH
                        The dictionary where the input is stored.
                        default:/home/lishaojie/data/cifar10/
  --job_dir JOB_DIR     The directory where the summaries will be stored.
                        default:./experiments
  --arch ARCH           Architecture of model. default:resnet
  --cfg CFG             Detail architecuture of model. default:resnet56
  --graph_gpu           Use gpu to graph the filters or not. default:False
```