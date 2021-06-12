import os  
import pathlib 
from tensorflow.keras.preprocessing import image_dataset_from_directory 
# -------------------------------------------------------------------------------------------------
dir = pathlib.Path('dataset2-master/images/TEST_SIMPLE')
data_test = image_dataset_from_directory(dir, label_mode='categorical', image_size=(150, 150), batch_size=30)
# -------------------------------------------------------------------------------------------------
 