# FasterR-CNN
The folder contains a FasterR-CNN implementation to localize nodules on breast ultrasound (US) dataset.

### Prerequisites
Running **train/predict** requires correct path to the input data and the following **packages** for ```python-3.x```

```
matplotlib==3.1.1
numpy==1.17.4
opencv-python==4.2.0
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
* Open ```predict.py``` and change the ```path_to_image``` variable to specify the location of the image for localizing nodule.
* Run the script from the editor or use ```python predict.py``` when running from terminal.

### Data Statistics

* Total Images in dataset = 644
* Number of Images for training = 516
* Number of Images for validation = 128

### Data Processing
* The ultrasoud images are normalized to a mean of and a variance of 0.5. The pixel values are scaled between 0 and 1.
* For the purpose of using CNNs with CUDA, the data was resized to a tensor size of [1, 256, 256].

### Model Parameter

* **Loss function**: Smooth-L1
* **Optimizer**: Adam
* **Epoch**: 200
* **Learning Rate**: 0.005 (reduces 1/10 if validation loss does not increase for 5 epochs) 

### Results

Model Name | Validation IoU 
--- | --- |
FasterR-CNN (COCO weights) | 0.7 |

#### Top Row (Ground-Truth) | Bottom Row (Model Predictions)
![Alt text](sample_prediction/prediction_1.png?raw=true "Sample Predictions")

##### When evaluated on a held-out test set, the IoU reported was 0.65.
