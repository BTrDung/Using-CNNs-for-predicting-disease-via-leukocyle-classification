import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.losses import CategoricalCrossentropy

from init_data_train import data_train

# for gpu in tf.config.experimental.list_physical_devices('GPU'):
#     tf.compat.v2.config.experimental.set_memory_growth(gpu, True)

# -------------------------------------------------------------------------------------
# cache if enough memory
# AUTOTUNE = tensorflow.data.AUTOTUNE
# data_train = data_train.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


model = tf.keras.Sequential()

model.add(Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block1_conv1',
                 input_shape=(150, 150, 3)))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block1_conv2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_maxpool'))

model.add(Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block2_conv1'))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block2_conv2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_maxpool'))

model.add(Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block3_conv1'))
model.add(Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block3_conv2'))
model.add(Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block3_conv3'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_maxpool'))

model.add(Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block4_conv1'))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block4_conv2'))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block4_conv3'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_maxpool'))

model.add(Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block5_conv1'))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block5_conv2'))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block5_conv3'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_maxpool'))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(4, activation='softmax'))


def optimizer_init_fn():
    learning_rate = 1e-4
    return tf.keras.optimizers.Adam(learning_rate)


model.compile(optimizer=optimizer_init_fn(),
              loss=CategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()
# ----------------------------------check point----------------
# from keras.callbacks import ModelCheckpoint, EarlyStopping
# checkpoint = ModelCheckpoint("vgg16_1.h5", verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# Fit model (training)
# history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
#                               validation_data=val_it, validation_steps=len(val_it), epochs=50, verbose=1)

# -------------------------------Train-------------------------
epoch = 6
history = model.fit(data_train, batch_size=30, epochs=epoch, verbose=1)
# model.fit(data_train, validation_data=val_ds, batch_size=30, epochs=30, verbose=1)

model.save("vgg.h5")


# ------------------------draw plt if use validation_data----------------------------
# draw_plt(history, epoch)
