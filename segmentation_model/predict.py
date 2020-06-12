from model_class import Resnet18Unet
import matplotlib.pyplot as plt
import cv2
from model_class import Resnet18Unet
import matplotlib.pyplot as plt

def get_mask(path_to_image=None, path_to_weights="checkpoints/Resnet18_UNet_0.19.pth"):
    if path_to_image:
        model_obj = Resnet18Unet(path_to_dict=path_to_weights, input_channel=1, num_classes=1, activation='sigmoid')
        prediction = model_obj.get_prediction(path_to_image)
        return prediction
    else:
        print("Image Path Required!")

if __name__ == "__main__":
    # WEIGHTS FOR MODEL
    weights_path = "checkpoints/Resnet18_UNet_0.19.pth"
    # PATH TO IMAGE
    path_to_image = "path_to_ultrasound_image\\" # (png/jpg/jpeg)
    path_to_image = "path_to_image (png/jpg/jpeg)"
    # RETURN A NUMPY ARRAY OF SIZE [256, 256]
    prediction = get_mask(path_to_image, weights_path)
    cv2.imwrite('prediction.png', prediction)
    print("PREDICTION SAVED!")
    plt.imshow(prediction)
    plt.show()


