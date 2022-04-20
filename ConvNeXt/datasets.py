# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

from sklearn.model_selection import KFold
from torch.utils.data import Dataset
import numpy as np
import random
import torch
from PIL import Image


def build_kfold_dataset(args):
    train_transforms = build_transform(is_train=True, args=args)
    val_transforms = build_transform(is_train=False, args=args)
    
    paths = [os.path.join(args.data_path, x) for x in os.listdir(args.data_path)]
    first = [os.path.join(args.data_path, '1', x) for x in os.listdir(args.data_path + '1')]
    third = [os.path.join(args.data_path, '3', x) for x in os.listdir(args.data_path + '3')]
    drova = [os.path.join(args.data_path, 'drova', x) for x in os.listdir(args.data_path + 'drova')]
    paths = np.array(first + third + drova)
    kf = KFold(n_splits=3, shuffle=True, random_state=args.seed)
    for i, (train_index, val_index) in enumerate(kf.split(paths)):
        if i + 1 == args.n_fold:
            train_paths, val_paths = paths[train_index], paths[val_index]
    train_dataset = WoodDataset(train_paths, train_transforms)
    val_dataset = WoodDataset(val_paths, val_transforms)
    return train_dataset, val_dataset
    
class WoodDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
        self.path2label = {'1': 1, 'drova': 0, '3': 2}
        self._len = len(self.paths)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        image_path = self.paths[index]
        image = Image.open(image_path)

        label_path = image_path.split('/')[-2]
        label = self.path2label[label_path]
        # using transform if necessary
        if self.transform:
            image = self.transform(image)
        return image, label

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
