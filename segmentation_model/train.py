from model_class import Resnet18Unet

path_to_weights = "checkpoints/Resnet18_UNet_0.19.pth"
# USED FOR TRAINING ONLY
path_to_images = "path_to_ultrasound_image (png/jpg/jpeg)"
path_to_masks = "path_to_ultrasound_masks (png/jpg/jpeg)"

if __name__ == "__main__":
    model_obj = Resnet18Unet(path_to_dict=path_to_weights, input_channel=1, num_classes=1, activation='sigmoid')
    # TO START TRAINING
    model_obj.train(path_to_images, path_to_masks)
