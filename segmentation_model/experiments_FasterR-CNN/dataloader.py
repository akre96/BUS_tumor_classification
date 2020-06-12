import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import os
from os import listdir
from torchvision import transforms
from numpy import clip
from skimage import io
from skimage.color import rgb2gray
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
    def preprocess(cls, img, new_size=False, expand_dim=False, adjust_label=False, normalize=False,
                   img_transforms=None):
        w, h = img.shape
        if new_size:
            assert new_size <= w or new_size <= h, 'Resize cannot be greater than image size'
            img = resize(img, (new_size, new_size))

        # Expand dimensions for image if specified
        if expand_dim is True:
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=0)
                img = img.transpose((2, 0, 1))

        if normalize:
            if img.max() > 1:
                img = (img - img.min()) / (img.max() - img.min())
            img = (img - img.mean()) / img.std()
            img = clip(img, -1.0, 1.0)
            img = (img + 1.0) / 2.0

        if adjust_label is True:
            coords = np.where(img != 0)
            img[coords] = 1

        if img_transforms:
            img = img_transforms(img)
        return img

    def __getitem__(self, i):
        img_idx = self.imgs_ids[i]
        mask_idx = self.mask_ids[i]
        img_file = self.imgs_dir + img_idx
        mask_file = self.masks_dir + mask_idx
        mask = rgb2gray(io.imread(mask_file).astype('float32'))
        img = rgb2gray(io.imread(img_file).astype('float32'))
        assert img.size == mask.size, \
            f'Image and mask {i} should be the same size, but are {img.size} and {mask.size}'
        img = self.preprocess(img, self.resize, expand_dim=False, adjust_label=False, normalize=True,
                              img_transforms=self.transformations)
        mask = self.preprocess(mask, self.resize, expand_dim=False, adjust_label=True, normalize=False,
                               img_transforms=self.transformations)

        # Computer bounding box
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        boxes = []
        for index in range(num_objs):
            pos = np.where(mask[index])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)  # tensor([[113., 153., 439., 382.]])
        labels = torch.ones((num_objs,), dtype=torch.int64)  # tensor([1])
        img_id = torch.tensor([i])  # tensor([1])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        # Set up annotations
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = area
        my_annotation["iscrowd"] = iscrowd
        return img, my_annotation

    def __len__(self):
        return len(self.imgs_ids)