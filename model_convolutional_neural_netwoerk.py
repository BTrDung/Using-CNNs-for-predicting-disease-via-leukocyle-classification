import collections
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

from init_data import xTrain, yTrain
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model

from PIL import Image

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
def plotFilters(conv_filter): 
    fig, axes = plt.subplots(1, 3, figsize = (5, 5)) 
    axes = axes.flatten() 
    for img, ax in zip(conv_filter, axes): 
        ax.imshow(img) 
        ax.axis('off') 
    plt.tight_layout()
    plt.show()

def visualize_all_filter(model): 
    for layer in model.layers:
        if 'conv' in layer.name:
            filters, bias= layer.get_weights()
            print(layer.name, filters.shape)
            #normalize filter values between  0 and 1 for visualization
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)  
            print(filters.shape[3])
            axis_x=1
            for i in range(filters.shape[3]): 
                filt=filters[:,:,:, i]
                plotFilters(filt)   
# -------------------------------------------------------------------------------------------------
model = tf.keras.models.Sequential([
    Convolution2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(150, 150, 3)), 
    MaxPooling2D(pool_size=(2, 2)), 
    Dropout(0.25),
    Convolution2D(filters=64, kernel_size=(3, 3), activation="relu"), 
    MaxPooling2D(pool_size=(2, 2)), 
    Dropout(0.25),
    Convolution2D(filters=128, kernel_size=(3, 3), activation="relu"), 
    MaxPooling2D(pool_size=(2, 2)), 
    Dropout(0.25),
    Flatten(), 
    Dense(128, activation ='relu'),
    Dropout(0.5),
    Dense(4, activation = "softmax")
]) 
model.summary() 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(xTrain, yTrain, batch_size = 5, epochs=3, verbose=1)
model.save("main.h5")
# ------------------------------------------------------------------------------------------------- 

model = load_model("main.h5")
