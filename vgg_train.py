import tensorflow as tf
import tensorflow.keras
from draw_plt_val_accur import draw_plt

from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib

# from init_data_train import data_train


# for gpu in tf.config.experimental.list_physical_devices('GPU'):
#     tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
# ---------------------split train and val------------------
train_data_dir = pathlib.Path('data2/5_class')
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   rotation_range=20,
                                   # zoom_range=0.2,
                                   vertical_flip=True,
                                   validation_split=0.2)  # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    subset='training',
    seed=20)  # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,  # same directory as training data
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    subset='validation')  # set as validation data
# -------------------------------------------------------------------------------------
# cache if enough memory
# AUTOTUNE = tensorflow.data.AUTOTUNE
# data_train = data_train.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


model = tf.keras.Sequential()

model.add(Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block1_conv1',
                 input_shape=(224, 224, 3)))
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
model.add(Dense(5, activation='softmax'))


def optimizer_init_fn():
    lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9)
    # learning_rate = 1e-4
    return tf.keras.optimizers.Adam(learning_rate=lr_schedule)


model.compile(optimizer=optimizer_init_fn(),
              loss=CategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()


# ----------------------------------check point----------------
# from keras.callbacks import ModelCheckpoint, EarlyStopping
# checkpoint = ModelCheckpoint("vgg16_1.h5", verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
# Fit model (training)
# history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
#                               validation_data=val_it, validation_steps=len(val_it), epochs=50, verbose=1)
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath='check_point/cp.h5',
#     save_weights_only=False,
#     monitor='val_accuracy',
#     mode='max',
#     save_best_only=True,
#     save_freq="epoch")
class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch == 40:  # or save after some epoch, each k-th epoch etc.
            self.model.save("vgg_epoch_{}.hd5".format(epoch))
        if epoch == 50:  # or save after some epoch, each k-th epoch etc.
            self.model.save("vgg_epoch_{}.hd5".format(epoch))


saver = CustomSaver()
callbacks_list = []
# -------------------------------Train-------------------------
epoch = 20
history = model.fit(train_generator, validation_data=validation_generator, batch_size=8, epochs=epoch, verbose=1, callbacks=callbacks_list)
# model.fit(data_train, validation_data=val_ds, batch_size=30, epochs=30, verbose=1)

model.save("vgg16.h5")

# ---------------continue-------------------------------------
loaded_model_h5 = tf.keras.models.load_model('vgg16.h5')
loaded_model_h5.fit()
# ------------------------draw plt if use validation_data----------------------------
draw_plt(history, epoch)
