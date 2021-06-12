import os  
import pathlib 
from tensorflow.keras.preprocessing import image_dataset_from_directory 
# -------------------------------------------------------------------------------------------------
dir = pathlib.Path('dataset2-master/images/TRAIN')
data_train = image_dataset_from_directory(dir, label_mode='categorical', image_size=(150, 150), batch_size=30)

# load val test
data_val_dir = pathlib.Path('dataset2-master/images/TEST')
# val_ds = image_dataset_from_directory(data_val_dir, label_mode='categorical', image_size=(150, 150), batch_size=30)

# -------------------------------------------------------------------------------------------------
 