import segmentation_models_pytorch as smp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from utils.dataloader import DataProcessor
import torch.optim as optim
from skimage.util import img_as_float, img_as_ubyte

class Resnet18Unet:
    def __init__(self, path_to_dict=None, input_channel=1, num_classes=1, activation='sigmoid'):
        self.path_to_dict = path_to_dict
        self.model = smp.Unet('resnet18', in_channels=input_channel, classes=num_classes, activation=activation,
                              encoder_weights='imagenet')
        if self.path_to_dict:
            weights = torch.load(path_to_dict)
            self.model.load_state_dict(weights)
        # Check for CUDA
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            self.device = torch.device("cpu")
            print("="*30)
            print("Running on CPU")
            print("=" * 30)
        else:
            print("=" * 30)
            self.device = torch.device("cuda:0")
            print("CUDA is available!")
            print("=" * 30)
        # Load model on CUDA
        self.model.to(self.device)

    def train_model(self, path_to_images=None, path_to_masks=None, transformation=None, val_percent=0.11, batch_size=5, lr_rate=0.001, num_epochs=200):
        # Data Loader
        if path_to_images and path_to_masks:
            dataset = DataProcessor(path_to_images, path_to_masks, transformations=transformation, resize_img=256)
            n_val = int(len(dataset) * val_percent)
            n_train = len(dataset) - n_val
            train, val = random_split(dataset, [n_train, n_val])
            print("Images for Training:", n_train)
            print("Images for Validation:", n_val)
            trainloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
            validloader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
            print("=" * 30)

            criterion = nn.BCELoss()
            optimizer = optim.RMSprop(self.model.parameters(), lr=lr_rate, weight_decay=1e-8, momentum=0.9)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

            valid_loss_min = np.Inf
            for epoch in range(num_epochs):
                # Define Constants
                train_loss = 0.0
                valid_loss = 0.0
                iou = 0.0
                epoch_loss = []
                for batch in trainloader:
                    data, target = batch['image'], batch['mask']
                    data, target = data.to(self.device, dtype=torch.float), target.to(self.device, dtype=torch.float)
                    optimizer.zero_grad()
                    output = self.model(data)
                    # Compute loss
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(float(loss.item() * data.size(0)))
                    train_loss += loss.item() * data.size(0)

                scheduler.step(np.mean(epoch_loss))
                with torch.no_grad():
                    self.model.eval()
                    for batch in validloader:
                        data, target = batch['image'], batch['mask']
                        data, target = data.to(self.device, dtype=torch.float), target.to(self.device, dtype=torch.float)
                        output = self.model(data)
                        loss = criterion(output, target)
                        valid_loss += loss.item() * data.size(0)
                        output_with_sigmoid = output
                        # Calculate IOU
                        true_lbl = img_as_ubyte(target.cpu().numpy())
                        convt_target = output_with_sigmoid.detach().cpu().numpy()
                        convt_mask = (convt_target > 0.5) * 255
                        pred_mask = convt_mask.astype(np.uint8)
                        iou += self._get_iou_vector(true_lbl, pred_mask)

                self.model.train()
                # Calculate average loss and accuracy
                train_loss = train_loss / len(trainloader)
                valid_loss = valid_loss / len(validloader)
                avg_iou = iou / len(validloader)
                print("Epoch:{}\t Training Loss:{:.6f}\t Validation Loss: {:.6f}\t Avg IoU: {:.6f}".format(epoch,
                                                                                                           train_loss,
                                                                                                           valid_loss,
                                                                                                           avg_iou))
                if valid_loss <= valid_loss_min:
                    print("Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(valid_loss_min,
                                                                                                    valid_loss))
                    print("-" * 40)
                    # Save model
                    torch.save(self.model.state_dict(), 'Resnet18_UNet.pth')
                    valid_loss_min = valid_loss
        else:
            return "Image and Mask path required!"

    def _get_iou_vector(self, target, prediction):
        run_iou = 0.0
        batch_size = target.shape[0]
        for index in range(batch_size):
            truth = target[index, 0]
            predicted = prediction[index, 0]
            intersection = np.logical_and(truth, predicted)
            union = np.logical_or(truth, predicted)
            run_iou += np.sum(intersection) / np.sum(union)
        run_iou /= batch_size
        return run_iou
