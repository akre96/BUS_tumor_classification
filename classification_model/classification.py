from pathlib import Path
from PIL import Image
import numpy as np
import os
from scipy.ndimage import label
from collections import Counter
from tensorflow.keras.models import load_model

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1*1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


def preprocess_mask(mask_arr):
    """
        Remove small segmentatopn region in the mask
    """

    # Get all region with positive mask
    labels, num = label(mask_arr)
    counter = Counter(labels.flatten())

    # Remove small mask region less than 200 px area
    for key in counter.keys():
        if key == 0:
            continue
        elif counter[key] > 200:
            labels[labels == key] = 255
        else:
            labels[labels == key] = 0
            
    return labels

def preprocess_input(img_path, mask_path, img_size=(224, 224)):
    """
            1. Remove small segmentation region in the mask
            2. Generate RGB image and resize to (224, 224) where
                Red channel: grayscale value of input image
                Green channel: segmentation mask
                Blue channel: blank


            Return RGB image size (224, 224, 3)
    """

    # Read image from path and add one more dimension
    img = Image.open(img_path)
    img_arr = np.array(img)
    if len(img_arr.shape) == 2:
        img_arr = np.expand_dims(img_arr, axis=-1)
    elif len(img_arr.shape) == 3:
        img_arr = img_arr[:, :, 0]
        img_arr = np.expand_dims(img_arr, axis=-1)

    # Read segmentation image from path and add one more dimension
    mask = Image.open(mask_path)
    mask_arr = np.array(mask)
    if len(mask_arr.shape) == 3:
        mask_arr = mask_arr[:, :, 0]
    final_mask_arr = preprocess_mask(mask_arr).astype('uint8')

    # No segmentation region found
    if np.sum(final_mask_arr) == 0:
        return None

    final_mask_arr = np.expand_dims(final_mask_arr, axis=-1)

    # Blank channel
    filler_arr = np.full(img_arr.shape, 0).astype('uint8')

    # Combine array to form RGB image: Red: input grayscale image, Green: segmentation mask image, Blue: blank array
    final_img_arr = np.concatenate((img_arr, final_mask_arr, filler_arr), axis=-1)
    img = Image.fromarray(final_img_arr).resize(img_size)
    img = np.array([np.array(img)])
    
    return img

def predict(img_path, mask_path, model_path):
    """
        Generate benign/malignant prediction from input image, mask and trained model
    """

    img = preprocess_input(img_path, mask_path)
    if img is None:
        return -1

    model = load_model(model_path, compile=False)
    y_pred = model.predict(img).flatten()[0]
    return y_pred
