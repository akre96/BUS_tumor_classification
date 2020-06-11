""" Example images from TorchGAN trained generator

Example usage:
```
python generate_conditional_gan_images.py [n_images] [output_directory]
```

Author: Samir Akre
"""
from pathlib import Path
import argparse
from tqdm import tqdm

import torch
import torchvision
from train_ConditionalGAN import dcgan_network
from torchgan.models import ConditionalGANGenerator




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("n_samples", type=int)
	parser.add_argument("output_dir")
	args = parser.parse_args()

	model_path = 'model/labeled_conditional_gan_goodfiles.model'

	model_config = dcgan_network["generator"]
	m_data = torch.load(model_path)
	generator = ConditionalGANGenerator(**model_config['args'])
	generator.load_state_dict(m_data['generator'])
	out_dir = Path(args.output_dir)


	# Seperate forlder for each class
	normal_dir = Path(out_dir, 'normal')
	benign_dir = Path(out_dir, 'benign')
	malignant_dir = Path(out_dir, 'malignant')
	status_dir = [normal_dir, benign_dir, malignant_dir]
	for d in status_dir:
		if d.is_dir():
			print('Directory exists, files of same name gen_{n}.png inside may be overwritter:', out_dir)
		else:
			print('Creating output directory:', d)
			d.mkdir()
		
	# Creates noise input and label for n_samples
	print('Generating', args.n_samples, 'samples...')
	noise_samples = generator.sampler(args.n_samples, 'cpu')

	# Generate image for each sample and save
	for i in tqdm(range(args.n_samples)):
		image = generator(noise_samples[0][i].unsqueeze(dim=0), noise_samples[1][i].unsqueeze(dim=0))
		label = noise_samples[1][i].tolist()
		file_name = Path(status_dir[label], 'label_' + str(label) + '_gen_' + str(i) + '.png')
		torchvision.utils.save_image(image, file_name)
