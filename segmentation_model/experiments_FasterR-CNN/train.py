from model_class import FasterRCNN

path_to_weights = "checkpoints/Breast_FRCNN_0.044.pth"
path_to_images = "path_to_ultrasound_image\\" # (png/jpg/jpeg)
path_to_masks = "path_to_ultrasound_masks\\" # (png/jpg/jpeg)

if __name__ == "__main__":
    model_obj = FasterRCNN(path_to_dict=path_to_weights, num_classes=2)
    # TO START TRAINING
    model_obj.train_model(path_to_images, path_to_masks, num_epochs=100)
