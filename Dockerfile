FROM tensorflow/tensorflow:latest-gpu

# Create a working directory
RUN mkdir /app
WORKDIR /app

RUN pip3 install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install segmentation-models-pytorch pyradiomics scikit-image scikit-learn Flask scipy Pillow
RUN pip3 install --upgrade Flask

RUN apt-get install -y git
RUN git clone https://github.com/akre96/BUS_tumor_classification.git
WORKDIR /app/BUS_tumor_classification
EXPOSE 5000
RUN chmod +x start_flask_dev.sh
RUN ls

# Set the default command to python3
ENTRYPOINT ["/bin/bash", "-c", "./start_flask_dev.sh"]
