# MIT License
#
# Copyright (c) 2019 Diana Wofk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Source: modified from https://github.com/dwofk/fast-depth
"""
import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py
from datasets import transforms

iheight, iwidth = 480, 640 # raw image size


def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth


class MyDataloader(data.Dataset):
    modality_names = ['rgb']

    def is_image_file(self, filename):
        IMG_EXTENSIONS = ['.h5']
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, dir, class_to_idx):
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self.is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
        return images

    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, split, modality='rgb', loader=h5_loader):
        root_path = os.getcwd()
        self.root = os.path.join(root_path, root)
        classes, class_to_idx = self.find_classes(self.root)
        imgs = self.make_dataset(self.root, class_to_idx)
        assert len(imgs)>0, "Found 0 images in subfolders of: " + self.root + "\n"
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        if split == 'train':
            self.transform = self.train_transform
        elif split == 'holdout':
            self.transform = self.val_transform
        elif split == 'val':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset split: " + split + "\n"
                                "Supported dataset splits are: train, val"))
        self.loader = loader

        assert (modality in self.modality_names), "Invalid modality split: " + modality + "\n" + \
                                "Supported dataset splits are: " + ''.join(self.modality_names)
        self.modality = modality

    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(self, rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    def __getraw__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path, target = self.imgs[index]
        rgb, depth = self.loader(path)
        return rgb, depth

    def __getitem__(self, index):
        rgb, depth = self.__getraw__(index)
        if self.transform is not None:
            rgb_np, depth_np = self.transform(rgb, depth)
        else:
            raise(RuntimeError("transform not defined"))

        if self.modality == 'rgb':
            input_np = rgb_np

        to_tensor = transforms.ToTensor()
        input_tensor = to_tensor(input_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)

        inputs = {}
        inputs[("color", 0, 0)] = input_tensor
        inputs["depth_gt"] = depth_tensor
        return inputs

    def __len__(self):
        return len(self.imgs)


class NYUDataset(MyDataloader):
    def __init__(self, root, split, modality='rgb', height=224, width=288):
        self.split = split
        super(NYUDataset, self).__init__(root, split, modality)
        self.output_size = (height, width)

    def is_image_file(self, filename):
        # IMG_EXTENSIONS = ['.h5']
        if self.split == 'train':
            return (filename.endswith('.h5') and \
                '00001.h5' not in filename and '00201.h5' not in filename)
        elif self.split == 'holdout':
            return ('00001.h5' in filename or '00201.h5' in filename)
        elif self.split == 'val':
            return (filename.endswith('.h5'))
        else:
            raise (RuntimeError("Invalid dataset split: " + self.split + "\n"
                                "Supported dataset splits are: train, val"))

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5) # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(240.0 / iheight), # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.HorizontalFlip(do_flip),
            transforms.CenterCrop(self.output_size),
        ])

        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize(240.0 / iheight),
            transforms.CenterCrop(self.output_size),
        ])

        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np