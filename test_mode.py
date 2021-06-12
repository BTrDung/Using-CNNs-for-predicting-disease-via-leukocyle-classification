import collections
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

from init_data_train import data_train
from init_data_test import data_test

from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("main.h5")
print(model.metrics_names)
print(model.evaluate(data_train, verbose=2))
# print(model.evaluate(data_test, verbose=2))