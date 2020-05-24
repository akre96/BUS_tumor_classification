""" Train a GAN on BUSI data set
Basic/test implementation of a GAN to generate
additional BUS images

Author: Samir Akre
"""
import random
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import Adam
from torchvision import transforms
from torchgan.models import (
    DCGANDiscriminator,
    DCGANGenerator
)
from torchgan.trainer import Trainer
from torchgan.losses import (
    LeastSquaresDiscriminatorLoss,
    LeastSquaresGeneratorLoss
)
from utils.busi_dataset import BUSI_Dataset

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
print("Random Seed: ", manualSeed)

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
	transform_mask=m_tfs
)


batch_size = 32
if torch.cuda.is_available():
    device = torch.device('cuda')
    epochs = 100

else:
    device = torch.device("cpu")
    epochs = 5

print('Device:', device)
print('Epochs:', epochs)
print('Batch Size:', batch_size)

dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

dcgan_network = {
    "generator": {
        "name": DCGANGenerator,
        "args": {
            "out_size": 256,
            "encoding_dims": 100,
            "out_channels": 1,
            "step_channels": 64,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.Tanh(),
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": DCGANDiscriminator,
        "args": {
            "in_size": 256,
            "in_channels": 1,
            "step_channels": 64,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.LeakyReLU(0.2),
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
    },
}

lsgan_losses = [LeastSquaresGeneratorLoss(), LeastSquaresDiscriminatorLoss()]
trainer = Trainer(
    dcgan_network, lsgan_losses, sample_size=64, epochs=epochs, device=device
)
trainer(dataloader)

