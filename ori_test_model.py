import collections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from tensorflow.keras.models import load_model
from PIL import Image
import os
import pathlib
from tensorflow.keras.preprocessing import image_dataset_from_directory
# -------------------------------------------------------------------------------------------------
dir = pathlib.Path('original_data/5_classes_test')
data_test = image_dataset_from_directory(dir, label_mode='categorical', image_size=(224, 224), batch_size=64)

# -------------------test VGG16--------------------------

model = load_model("vgg16.h5")

# -------------------test VGG19---------------------------

# model = load_model("vgg19.h5")

# -------------------test Resnet50------------------------

# model = load_model("resnet50.h5")

# ----------------------------------------------------------
# print(model.metrics_names)
model.summary()
print('Data: TEST')
print(print(model.evaluate(data_test, verbose=2)))
# ---------------------------------------------
