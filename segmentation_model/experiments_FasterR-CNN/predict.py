from model_class import FasterRCNN
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray

import cv2

def get_mask(path_to_image=None, path_to_weights="checkpoints/Breast_FRCNN_0.044.pth"):
    if path_to_image:
        model_obj = FasterRCNN(path_to_dict=path_to_weights, num_classes=2)
        prediction = model_obj.get_prediction(path_to_image)
        return prediction
    else:
        print("Image Path Required!")

if __name__ == "__main__":
    # WEIGHTS FOR MODEL
    weights_path = "checkpoints/Breast_FRCNN_0.044.pth"
    # PATH TO IMAGE
    path_to_image = "path_to_ultrasound_image\\" # (png, jpg, jpeg)
    # RETURN BBOX
    bbox = get_mask(path_to_image, weights_path)
    if len(bbox) == 0:
        print("No Bounding Box Detected")
    else:
        image = rgb2gray(io.imread(path_to_image))
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(img_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)
        cv2.imwrite('prediction.png', img_rgb)
        print("PREDICTION SAVED!")
        plt.imshow(img_rgb)
        plt.show()

