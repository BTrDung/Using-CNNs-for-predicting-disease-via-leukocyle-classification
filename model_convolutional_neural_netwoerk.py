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

from PIL import Image

# -------------------------------------------------------------------------------------------------
def get_layers(all_weights, color, rows, columns, layer): 
    # [0][color][row][column][layer] - Layer should be subtracted by 1. 
    filter_layer = [] 
    print(all_weights[0])
    for row in range(0, rows): 
        col = [] 
        for column in range(0, columns): 
            col.append(all_weights[0][color][row][column][layer].numpy())
        filter_layer.append(col)
    return filter_layer
    
# -------------------------------------------------------------------------------------------------
def combine(r, g, b): 
    rows = len(r) 
    columns = len(r[0])
    arr = [] 
    for row in range(0, rows): 
        sub = [[r[row][column], g[row][column], b[row][column]] for column in range(0, columns)]
        arr.append(sub)
    print('Converting a array to image...')
    arr = np.array(arr)
    img = Image.fromarray(arr, 'RGB')
    img.show()
    return None
# -------------------------------------------------------------------------------------------------
def get_conv(filter, color, org_img): 
    img     = np.copy(org_img) 
    rows    = img.shape[1] 
    columns = img.shape[2] 
    arr     = [[0 for col in range(0, columns - (len(filter) - 1))] for row in range(0,rows - (len(filter) - 1))]
    for row in range(0, rows - (len(filter) - 1)): 
        for column in range(0, columns - (len(filter) - 1)): 
            value = 0.0 
            for i in range(0, len(filter)): 
                for j in range(0, len(filter)): 
                    value = value + (img[0][row + i][column + j][color] * filter[i][j])
            # value = img[0][row][column][color] 
            arr[row][column] = value
    return arr
# -------------------------------------------------------------------------------------------------
def remake_img(filters, org_img): 
    img = np.copy(org_img) 
    print('Getting RED color filter...')
    r   = get_conv(filters[0] ,0 ,org_img)
    # m = np.copy(r) 
    # m = Image.fromarray(m, 'L')
    # m.show()
    print('Getting BLUE color filter...')
    g   = get_conv(filters[1] ,1 ,org_img)
    print('Getting GREEN color filter...')
    b   = get_conv(filters[2] ,2 ,org_img)
    rbg = combine(r, g, b) 
    return None
# -------------------------------------------------------------------------------------------------
def getImg(link): 
    with open(link, 'rb') as f: 
        img = Image.open(f)
        img = img.convert('RGB')
        # img.show()
        return np.array(img)
    return None 
# -------------------------------------------------------------------------------------------------
model = tf.keras.models.Sequential([
    Convolution2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(240, 320, 3)), 
    MaxPooling2D(pool_size=(2, 2)), 
    Convolution2D(filters=64, kernel_size=(3, 3), activation="relu"), 
    MaxPooling2D(pool_size=(2, 2)), 
    Convolution2D(filters=128, kernel_size=(3, 3), activation="relu"), 
    MaxPooling2D(pool_size=(2, 2)), 

    Flatten(), 
    Dense(4, activation = "softmax")
]) 
model.summary() 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# ------------------------------------------------------------------------------------------------- 
model.fit(xTrain, yTrain, batch_size = 5, epochs=100, verbose=1)
filter_layers =[get_layers(model.layers[0].weights, color=0,rows=3, columns=3, layer=0), 
                get_layers(model.layers[0].weights, color=1,rows=3, columns=3, layer=0), 
                get_layers(model.layers[0].weights, color=2,rows=3, columns=3, layer=0)]

# remake_img(filter_layers, img)
# model.fit(xTrain, yTrain, batch_size=1, validation_split=0.2, epochs=5)
# model_ = model.layers[:4]
# # print(model.layers[2].weights)
# print(model.predict(img))
# print(get_layers(model.layers[0].weights, color=0,rows=3, columns=3, layer=0))
# -----------------------------------------------
