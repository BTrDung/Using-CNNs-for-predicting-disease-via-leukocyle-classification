import collections
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

from init_data_train import data_train
# from init_data_train import val_ds
from init_data_test import data_test

from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("adam100.h5")
np.seterr(divide='ignore', invalid='ignore')

successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

img = tf.keras.preprocessing.image.load_img('.\dataset2-master\images\TEST\EOSINOPHIL\_0_187.jpeg', target_size=(150, 150))
x   = tf.keras.preprocessing.image.img_to_array(img) 
x   = x.reshape((1, ) + x.shape)
x   = x / 255.0 

successive_feature_maps = visualization_model.predict(x)
layer_names = [layer.name for layer in model.layers]


count = 0
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    print(feature_map.shape)
    if len(feature_map.shape) == 4:
        #-------------------------------------------
        # Just do this for the conv / maxpool layers, not the fully-connected layers
        #-------------------------------------------
        n_features = feature_map.shape[-1]  # number of features in the feature map
        size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
        
        # We will tile our images in this matrix
        display_grid = np.zeros((size, size * n_features))
        
        for i in range(n_features):
            x  = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std ()
            x *=  64
            x += 128
            x  = np.clip(x, 0, 255).astype('uint8')
            # Tile each filter into a horizontal grid
            display_grid[:, i * size : (i + 1) * size] = x 

        #-----------------
        # Display the grid
        #-----------------
        scale = 20. / n_features
        plt.figure( figsize=(scale * n_features, scale) )
        plt.title ( layer_name )
        plt.grid  ( False )
        plt.imshow( display_grid, aspect='auto', cmap='viridis')
        # plt.show()
        count += 1
        plt.savefig(str(count) + '.png')