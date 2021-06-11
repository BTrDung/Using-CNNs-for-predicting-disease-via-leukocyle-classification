import collections
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.layers import Dropout

# from init_data import xTrain, yTrain
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
    img = np.copy(org_img)
    rows = img.shape[1]
    columns = img.shape[2]
    arr = [[0 for col in range(0, columns - (len(filter) - 1))] for row in range(0, rows - (len(filter) - 1))]
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
    r = get_conv(filters[0], 0, org_img)
    # m = np.copy(r)
    # m = Image.fromarray(m, 'L')
    # m.show()
    print('Getting BLUE color filter...')
    g = get_conv(filters[1], 1, org_img)
    print('Getting GREEN color filter...')
    b = get_conv(filters[2], 2, org_img)
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
#----
import pathlib
data_dir = pathlib.Path('images/TRAIN_SIMPLE')
dataset = tf.keras.preprocessing.image_dataset_from_directory(data_dir, label_mode='categorical', image_size=(150, 150), batch_size=30)
data_dir = pathlib.Path('images/TEST')
val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,label_mode='categorical', image_size=(150, 150), batch_size=30)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = dataset.cache().prefetch(buffer_size=AUTOTUNE)


# normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# normalized_dsval = val_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))

#----
model = tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(150, 150, 3)),
    # Convolution2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(240, 320, 3)),
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
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# -------------------------------------------------------------------------------------------------
# model.fit(xTrain, yTrain, batch_size=30, epochs=30, verbose=2)
model.fit(dataset, epochs=30, verbose=1)

model.save("main_load_by_keras.h5")

print(datetime.datetime.now())
