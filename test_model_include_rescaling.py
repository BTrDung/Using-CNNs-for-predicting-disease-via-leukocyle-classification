import collections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# from init_data import xTrain, yTrain
# from init_data_test import xTest, yTest

from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from tensorflow.keras.models import load_model
from PIL import Image
import pathlib
data_dir = pathlib.Path('images/TEST_SIMPLE')
datatest = tf.keras.preprocessing.image_dataset_from_directory(data_dir, label_mode='categorical', image_size=(150, 150), batch_size=30)
data_dir = pathlib.Path('images/TEST')
datatestsimple = tf.keras.preprocessing.image_dataset_from_directory(data_dir, label_mode='categorical', image_size=(150, 150), batch_size=30)

model = load_model("main_load_by_keras.h5")
print(model.metrics_names)
# print(model.evaluate(xTrain, yTrain, verbose=2))
# print(model.evaluate(xTest, yTest, verbose=2))
print(model.evaluate(datatestsimple, verbose=2))
print(model.evaluate(datatest, verbose=2))

