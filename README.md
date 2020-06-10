# Breast Ultrasound Image Tumor Segmentation and Classification
A Project for BE 223C


## File Descriptions
- __train_gan.py:__ Trains a DCGAN based network to create fake ultrasound images
- __flask_app.py:__ Web application for UI of segmentation/classification pipeline
- __start_flask_dev.sh:__ Starts the flask web application in development mode to port 5000
- __utils/__: Functions for loading/processing of data
	- __busi_dataset.py__: Loads BUSI image data set for use with pytorch
- __templates/__: Jinja2 templates for flask app
- __static/__: Flask app static files (images, css, etc.)

