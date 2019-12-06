"""Dataset.
"""

import os
import pandas as pd
import torch
from PIL import Image
import numpy as np
from torchvision import get_image_backend


class Datasets(torch.utils.data.Dataset):
    """Dataset.
    """
    def __init__(self, train=True, transform=None, iter_no=1):
        self.train = train
        self.transform = transform
        data_dir = '../data/'
        self.train_data, self.test_data = self.make_dataset(iter_no, data_dir)

        images_data = 'ISIC2018_Task3_Training_Input'
        self.data_dir = data_dir
        self.mask_dir = os.path.join(self.data_dir, 'masks', '{}_mask')
        self.images_dir = os.path.join(data_dir, images_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the
                   target class.
        """

        if self.train:
            path, target = self.train_data[index]
        else:
            path, target = self.test_data[index]

        img_path = os.path.join(self.images_dir, path)
        img_name = path.split('.')[0]
        imagedata = self.default_loader(img_path)
        length = 224

        if self.transform is not None:
            imagedata = self.transform(imagedata)

        if self.train:
            mask_dir = self.mask_dir.format(target)
            if os.path.exists(mask_dir):
                has_mask = True
                inner_path = os.path.join(mask_dir, img_name+"_inner.npy")
                outer_path = os.path.join(mask_dir, img_name+"_outer.npy")
                inner = np.load(inner_path).astype(np.float32)
                outer = np.load(outer_path).astype(np.float32)
            else:
                has_mask = False
                inner = np.zeros((length, length), dtype=np.float32)
                outer = np.zeros((length, length), dtype=np.float32)

            return imagedata, target, has_mask, inner, outer
        else:
            return imagedata, target

    def __len__(self):
        """len.
        """
        if self.train:
            length = len(self.train_data)
        else:
            length = len(self.test_data)
        return length

    def make_dataset(self, iter_no, data_dir):
        """make_dataset
        Args:
            iter_no
        """
        train_csv = 'split_data/split_data_{}_fold_train.csv'.format(iter_no)
        train_csv = os.path.join(data_dir, train_csv)

        csvfile = pd.read_csv(train_csv, index_col=0)
        raw_train_data = csvfile.values

        train_data = []
        for img_name, label in raw_train_data:
            train_data.append((img_name, label))

        test_csv = 'split_data/split_data_{}_fold_test.csv'.format(iter_no)
        test_csv = os.path.join(data_dir, test_csv)
        print('Loading train from {}'.format(test_csv))

        csvfile = pd.read_csv(test_csv, index_col=0)
        raw_test_data = csvfile.values

        test_data = []
        for img_name, label in raw_test_data:
            test_data.append((img_name, label))

        return train_data, test_data

    def accimage_loader(self, path):
        """accimage_loader"""
        try:
            import accimage
            return accimage.Image(path)
        except ImportError:
            return self.pil_loader(path)

    def pil_loader(self, path):
        """pil_loader
        Args:
            path: path to load image
        Return:
            image
        """
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def default_loader(self, path):
        """default_loader
        Args:
            path: image path to load
        Return:
            image
        """
        if get_image_backend() == 'accimage':
            img = self.accimage_loader(path)
        else:
            img = self.pil_loader(path)
        return img
