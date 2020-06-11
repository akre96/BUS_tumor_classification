""" Example images from TorchGAN trained generator

Example usage: 
```
python generate_dcgan_images.py [n_images] [output_directory]
```
Author: Samir Akre
"""
from pathlib import Path
import argparse
from tqdm import tqdm

import torch
import torchvision
from train_DCGAN import dcgan_network
from torchgan.models import DCGANGenerator




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("n_samples", type=int)
	parser.add_argument("output_dir")
	args = parser.parse_args()

	model_path = 'model/labeled_gan_goodfiles.model'

	model_config = dcgan_network["generator"]
	m_data = torch.load(model_path)
	generator = DCGANGenerator(**model_config['args'])
	generator.load_state_dict(m_data['generator'])
	out_dir = Path(args.output_dir)
	if out_dir.is_dir():
		print('Directory exists, files of same name gen_{n}.png inside may be overwritter:', out_dir)
	else:
		print('Creating output directory:', out_dir)
		out_dir.mkdir()
	print('Generating', args.n_samples, 'samples...')
	noise_samples = generator.sampler(args.n_samples, 'cpu')
	images = generator(noise_samples[0])
	for i, image in tqdm(enumerate(images)):
		file_name = Path(out_dir, 'gen_' + str(i) + '.png')
		torchvision.utils.save_image(image, file_name)
