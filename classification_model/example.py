from pathlib import Path
import os
from classification import predict

data_dir = Path('../segmentation_model/sample_predictions')
img_dir = data_dir / 'images'
mask_dir = data_dir / 'mask_predicted'

img_list = os.listdir(img_dir)
img_list.sort()

mask_list = os.listdir(mask_dir)
mask_list.sort()

i = 0
model_path = Path('./models/model.h5')
y_pred = predict(img_dir / img_list[i], mask_dir / mask_list[i], model_path)
print(y_pred)