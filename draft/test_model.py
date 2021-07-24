import numpy as np
import tensorflow as tf

# from init_data_train import val_ds

from tensorflow.keras.models import load_model

# ---------------------------------------------

model = load_model("adam100.h5")
print(model.metrics_names)
img = tf.keras.preprocessing.image.load_img('dataset2-master/images/TEST_SIMPLE/LYMPHOCYTE/1.jpeg', target_size=(150, 150))
img = tf.keras.preprocessing.image.img_to_array(img)
img = np.array([img])
print(model.predict(img))
model.summary()
# print('Data: TRAIN')
# print(model.evaluate(data_train, verbose=2))

# print('Data: TEST')
# print(print(model.evaluate(data_test, verbose=2)))

# print('Data: TEST_SIMPLE')
# print(model.evaluate(data_test, verbose=2))
# ---------------------------------------------
