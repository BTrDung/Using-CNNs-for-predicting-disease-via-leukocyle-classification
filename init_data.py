import os
import collections
import numpy as np 
import tensorflow as tf 

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array 
from tensorflow.keras.preprocessing.image import array_to_img 
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
  
from PIL import Image
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
link_data_train = 'E:/SelfStudyTryHard/CS213/dataset2-master/images/TRAIN'
xTrain = []
yTrain = []
count = 0
for name_data in os.listdir(link_data_train): 
    for img in os.listdir(link_data_train + '/' + name_data): 
        image = load_img(link_data_train + '/' + name_data + '/' + img, target_size=(150, 150))
        image = img_to_array(image)
        xTrain.append(image) 
        yTrain.append([count])
    count += 1
# -------------------------------------------------------------------------------------------------

xTrain = np.array(xTrain) 
yTrain = np.array(yTrain)

xTrain = xTrain / 255 
xTrain = xTrain.reshape(len(xTrain), 150, 150, 3)

