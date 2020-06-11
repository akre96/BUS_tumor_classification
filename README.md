# Breast Ultrasound Image Tumor Segmentation and Classification
A Project for BE 223C

## Usage Instructions

### Building and running from Docker image
1. Select a dockerfile to use, if GPU access is avilable and configured (with nvidia-docker installed) use `Dockerfile-gpu` otherwise use `Dockerfile-cpu`
2. To build the image for CPU run:
```
docker build -t bus_seg:cpu -f Dockerfile-cpu .
```

To build the image for GPU: 
Follow instruction [here](https://github.com/NVIDIA/nvidia-docker/blob/master/README.md#quickstart) to install nvidia docker toolkit then run:
```
docker build -t bus_seg:gpu -f Dockerfile-gpu .
```
3. Start the built image, the Flask application will be running on port 5000
```
## For CPU
docker run --publish=5000:5000 bus_seg:cpu

## For GPU
docker run --gpus all --publish=5000:5000 bus_seg:gpu

```
4. The running app can now be accessed from `127.0.0.1:5000` on the local machine

## File Descriptions

### TorchGAN
_Note:_ The below scripts assume they are being run in the AWS instance as the trained models are not uploaded on github. Path to trained models: `/home/sakre/BUS_tumor_classification/GAN/model`  

- __GAN/:__ Folder containing training/usage scripts for TorchGAN attempts on BUSI dataset
	- __train_ACGAN.py:__ Trains an Auxillary Classifier GAN
	- __train_ConditionalGAN.py:__ Trains a Conditional GAN
	- __train_DCGAN.py:__ Trains a standard Deep Convolutional GAN
	- __generate_conditional_gan_images.py:__ Generates example images from conditional GAN model trained for 2k epochs
	- __generate_dcgan_images.py:__ Generates example images from DCGAN model trained for 2k epochs

### Flask App
- __flask_app.py:__ Web application for UI of segmentation/classification pipeline
- __start_flask_dev.sh:__ Starts the flask web application in development mode to port 5000
- __templates/__: Jinja2 templates for flask app  
	- __base.html__: Imports bootstrap style sheets, JQuery, and BootstrapJS 
	- __home.html__: Main page, loads elements from `image_viewer.html` and `menu.html`
	- __image_viewer.html__: Image component. Handles overlay of mask/BUS image
	- __menu.html__: Contains all buttons/forms used
- __static/__: Flask app static files (images, css, etc.)

### Utilities
- __utils/__: Functions for loading/processing of data
	- __busi_dataset.py__: Loads BUSI image data set for use with pytorch
