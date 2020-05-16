import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import os
from os import listdir
from torchvision import transforms
from numpy import clip
from skimage.util import img_as_float, img_as_ubyte
from skimage.transform import resize

class DataProcessor(Dataset):
    def __init__(self, imgs_dir, masks_dir, transformations=None, resize_img=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transformations = transformations
        self.resize = resize_img
        self.imgs_ids = [file for file in listdir(imgs_dir)]
        self.mask_ids = [file for file in listdir(masks_dir)]

    @classmethod
    def preprocess(cls, img, new_size=False, expand_dim=False, adjust_label=False, normalize=False, img_transforms=None):
        w, h = img.shape
        if new_size:
            assert new_size <= w or new_size <= h, 'Resize cannot be greater than image size'
            img = resize(img, (new_size, new_size))
        # Expand dimensions for images if specified
        if expand_dim is True:
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)

        img_trans = img.transpose((2, 0, 1))
        # Scale pixel values
        if img_trans.max() > 1:
            img_trans = (img_trans - img_trans.min()) / (img_trans.max() - img_trans.min())

        # Standarize pixel values
        if normalize:
            img_trans = (img_trans - img_trans.mean()) / img_trans.std()
            img_trans = clip(img_trans, -1.0, 1.0)
            img_trans = (img_trans + 1.0) / 2.0

        # For mask that has values between 0 and 1
        if adjust_label is True:
            coords = np.where(img_trans != 0)
            img_trans[coords] = 1

        # Apply transformations if specified
        if img_transforms:
            img_trans = img_transforms(img_trans)
        return img_trans

    def __getitem__(self, i):
        img_idx = self.imgs_ids[i]
        mask_idx = self.mask_ids[i]
        img_file = self.imgs_dir + img_idx
        mask_file = self.masks_dir + mask_idx
        # Read data (expects numpy here, change accordingly)
        mask = np.load(mask_file).astype('float32')
        img = np.load(img_file).astype('float32')
        assert img.size == mask.size, \
            f'Image and mask {i} should be the same size, but are {img.size} and {mask.size}'
        img = self.preprocess(img, self.resize, expand_dim=True, adjust_label=False, normalize=True,
                              img_transforms=self.transformations)
        mask = self.preprocess(mask, self.resize, expand_dim=True, adjust_label=True, normalize=False,
                               img_transforms=self.transformations)
        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

    def __len__(self):
        return len(self.imgs_ids)