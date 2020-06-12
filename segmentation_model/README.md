# Resnet18-UNet
The folder contains a UNet model with Resent18 backbone, trained on breast ultrasound (US) dataset for 3 class (normal, malignant, benign) nodule segmentation.

## Directory Structure
```
checkpoints\ -> Saved weights for the model (used for Inference)
sample_predictions\ -> Contains some of the prediction examples from this model
```
### Prerequisites
Running **train/predict** requires correct path to the input data and the following **packages** for ```python-3.x```

```
matplotlib==3.1.1
numpy==1.17.4
scikit-image==0.15.0
segmentation-models-pytorch==0.1.0
torch==1.4.0+cu92
torchvision==0.5.0+cu92
```

#### Running Training Script
* Open ```train.py``` and change the ```path_to_images``` and ```path_to_masks``` variable to specify the location of training images 
and their respective masks respectively.
* Run the script from the editor or use ```python train.py``` when running from terminal.

#### Running Prediction Scripts
* Open ```predict.py``` and change the ```path_to_image``` variable to specify the location of the image for mask segmentation.
* Run the script from the editor or use ```python predict.py``` when running from terminal.

### Data Statistics

* Total Images in dataset = 644
* Number of Images for training = 516
* Number of Images for validation = 128

### Data Processing
* The ultrasoud images are normalized to a mean of and a variance of 0.5. The pixel values are scaled between 0 and 1.
* The number of US images are upsampled by 7 times -> 7 x 644 = 4508 variations (Train/Val -> 3607/901).
* For the purpose of using CNNs with CUDA, the data was resized to a tensor size of [1, 256, 256].

### Model Parameter

* **Loss function**: Pixel-wise Binary Cross Entropy
* **Optimizer**: RMSProp
* **Epoch**: 200
* **Learning Rate**: 0.001 (reduces 1/10 if validation loss does not increase for 5 epochs) 

### Results

#### Different variations of UNet 
Model Name | Validation Loss 
--- | --- |
UNet (No pretrained weights) | 0.9 |
Resnet18-UNet (ImageNet weights) | 0.5 |
Resnet50-UNet (ImageNet weights) | 0.4 |
Densenet121-UNet (ImageNet weights) | 0.3 |
Resnet18-Unet (retrained with previous weights) | 0.1 |

##### Higest Validation Iou (0.75) was acheived by Resnet18-UNet on an held-out external dataset.

## Loss Graph
#### UNet | Resnet18-Unet
![Alt text](loss_graph/U-netLoss_training_1.png?raw=true "Title")
![Alt text](loss_graph/Resnet_18_Unet_loss_training_3.png?raw=true "UNet")

#### Resnet50-UNet | Densenet121-Unet
![Alt text](loss_graph/Resnet_50_Unet-loss_training.png?raw=true "UNet")
![Alt text](loss_graph/Densenet121_Unet-loss_training.png?raw=true "UNet")


## Conclusion
* Out of the 3 classes, the **model** demonstrates reasonable performance in identifying **benign nodules**. It is harder to detect malignant cases because of the given complexity in shape and size of malignant nodules.
A different segmentation model that can account for the nodule location as well as its shape & size could yield in better segmentation.