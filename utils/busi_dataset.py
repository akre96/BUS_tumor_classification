""" Loads BUSI breast ultrasound image data set

Use in conjunction with a pytorch dataloader to train/test models

Author: Samir Akre
"""
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np

class BUSI_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        transform=None,
        transform_mask=None,
    ):
        self.masks = list(Path(root_dir).glob('**/*_mask.png'))
        self.imgs = [
            Path(m.parent, m.name.replace('_mask', ''))
            for m in self.masks
        ]
        self.labels = [
            im.name.split(' ')[0] for im in self.imgs
        ]
        self.label_encode_dict = {
            'normal': 0,
            'benign': 1,
            'malignant': 2,
        }
        self.label_encoded = [
            self.label_encode_dict[l] for l in self.labels
        ]
        self.transform = transform
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.label_encoded)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = Image.open(self.imgs[idx]).convert('L')
        mask = Image.open(self.masks[idx]).convert('L')
        label = torch.tensor([self.label_encoded[idx]], dtype=torch.long)
        if self.transform:
            img = self.transform(img)
            if self.transform_mask:
                mask = self.transform_mask(mask)

        return img, label, mask

