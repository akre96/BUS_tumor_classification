from model_class import Resnet18Unet
import matplotlib.pyplot as plt

path_to_weights = "checkpoints/Resnet18_UNet_0.19.pth"
# USED FOR INFERENCE ONLY
path_to_image = "path_to_image (png/jpg/jpeg)"

if __name__ == "__main__":
    model_obj = Resnet18Unet(path_to_dict=path_to_weights, input_channel=1, num_classes=1, activation='sigmoid')
    # GET PREDICTION
    prediction = model_obj.get_prediction(path_to_image)
    print(type(prediction))
    print(prediction.shape)
    plt.imshow(prediction)
    plt.show()

