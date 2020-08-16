import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

class Data:

    def __init__(self, args):
        pin_memory = False
        if args.gpus is not None:
            pin_memory = True

        traindir = os.path.join(args.data_path, 'ILSVRC2012_img_train')
        valdir = os.path.join(args.data_path, 'val')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        trainset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4,
                    saturation=0.4),
                Lighting(0.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                
            ]))

        self.trainLoader = DataLoader(
            trainset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=pin_memory)

        testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        self.testLoader = DataLoader(
            testset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=10,
            pin_memory=True)

imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Lighting(object):
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'
