""" Train a GAN on BUSI data set
Basic/test implementation of a GAN to generate
additional BUS images

Author: Samir Akre
"""
import sys
sys.path.insert(0, '..utils')

import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
from torchvision import transforms
from torchgan.models import (
    ConditionalGANDiscriminator,
    ConditionalGANGenerator
)
from torchgan.trainer import Trainer
from torchgan.losses import (
    LeastSquaresDiscriminatorLoss,
    LeastSquaresGeneratorLoss
)

from busi_dataset import BUSI_Dataset

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
print("Random Seed: ", manualSeed)

dcgan_network = {
    "generator": {
        "name": ConditionalGANGenerator,
        "args": {
            "out_size": 256,
            "encoding_dims": 128,
            "out_channels": 1,
            "step_channels": 128,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.Tanh(),
	    "num_classes": 3
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": ConditionalGANDiscriminator,
        "args": {
            "in_size": 256,
            "in_channels": 1,
            "step_channels": 128,
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

	batch_size = 32
	if torch.cuda.is_available():
	    device = torch.device('cuda')
	    epochs = 2000

	else:
	    device = torch.device("cpu")
	    epochs = 5

	print('Device:', device)
	print('Epochs:', epochs)
	print('Dataset Total Size:', dataset_size)
	print('Dataset For Validation Remaining:', split)
	print('Batch Size:', batch_size)

	dataloader = data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)


	lsgan_losses = [LeastSquaresGeneratorLoss(), LeastSquaresDiscriminatorLoss()]
	trainer = Trainer(
	    dcgan_network, lsgan_losses, sample_size=64, epochs=epochs, device=device
	)
	trainer(dataloader)

