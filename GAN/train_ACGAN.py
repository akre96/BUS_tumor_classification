""" Train a GAN on BUSI data set
Basic/test implementation of a GAN to generate
additional BUS images

Author: Samir Akre
"""
import sys
from pathlib import Path
sys.path.insert(0, '../utils')

import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
from torchvision import transforms
from torchgan.models import (
    ACGANDiscriminator,
    ACGANGenerator
)
from torchgan.trainer import Trainer
from torchgan.losses import (
    AuxiliaryClassifierDiscriminatorLoss,
    AuxiliaryClassifierGeneratorLoss
)

from busi_dataset import BUSI_Dataset

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
print("Random Seed: ", manualSeed)

dcgan_network = {
    "generator": {
        "name": ACGANGenerator,
        "args": {
            "out_size": 256,
            "encoding_dims": 128,
            "out_channels": 1,
            "step_channels": 64,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.Tanh(),
	    "num_classes": 3
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": ACGANDiscriminator,
        "args": {
            "in_size": 256,
            "in_channels": 1,
            "step_channels": 64,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.LeakyReLU(0.2),
	    "num_classes": 3
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
    },
}
if __name__ == '__main__':
	tfs = transforms.Compose(
		[
		    transforms.Resize((256, 256)),
		    transforms.ToTensor(),
		    transforms.Normalize(mean=(0.5,), std=(0.5,)),
		]
	)
	m_tfs = transforms.Compose(
		[
		    transforms.Resize((256, 256)),
		    transforms.ToTensor(),
		]
	)
	dataset = BUSI_Dataset(
		'/data/dataset_busi',
		transform=tfs, 
		transform_mask=m_tfs,
	    goodfiles_only=True
	)

	validation_split = .2
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	split = int(np.floor(validation_split * dataset_size))
	np.random.seed(manualSeed)
	np.random.shuffle(indices)
	train_indices, valid_indices = indices[split:], indices[split:]

	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(valid_indices)

	batch_size = 32*2
	if torch.cuda.is_available():
	    device = torch.device('cuda')
	    epochs = 4000

	else:
	    device = torch.device("cpu")
	    epochs = 5

	print('Device:', device)
	print('Epochs:', epochs)
	print('Dataset Total Size:', dataset_size)
	print('Dataset For Validation Remaining:', split)
	print('Batch Size:', batch_size)
	
	checkpoint_prefix = './model/ACGAN_goodfiles'
	print('Checkpoint Prefix:', checkpoint_prefix)

	images_dir = Path('./sample_images')
	if not images_dir.is_dir():
		images_dir.mkdir()
	print('Samples image dir:', images_dir)

	log_dir = Path('./log/ACGAN')
	if not log_dir.is_dir():
		log_dir.mkdir()
	print('Log dir:', log_dir)

	dataloader = data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

	lsgan_losses = [AuxiliaryClassifierGeneratorLoss(), AuxiliaryClassifierDiscriminatorLoss()]
	trainer = Trainer(
		dcgan_network,
		lsgan_losses,
		sample_size=64,
		epochs=epochs,
		device=device, 
		checkpoints=checkpoint_prefix,
		recon=images_dir,
		log_dir=log_dir,
	)

	print('Continueing from epoch 2000')
	trainer.load_model('model/ACGAN_goodfiles4.model')
	trainer(dataloader)

