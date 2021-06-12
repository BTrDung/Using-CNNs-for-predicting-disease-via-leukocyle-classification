import os  
import pathlib 
from tensorflow.keras.preprocessing import image_dataset_from_directory 
# -------------------------------------------------------------------------------------------------
dir = pathlib.Path('E:/SelfStudyTryHard/CS213/dataset2-master/images/TRAIN')
data_train = image_dataset_from_directory(dir, label_mode='categorical', image_size=(150, 150), batch_size=30)
# -------------------------------------------------------------------------------------------------
 