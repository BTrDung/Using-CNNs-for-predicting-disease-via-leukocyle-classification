# from init_data_train import val_ds

from tensorflow.keras.models import load_model

# ---------------------------------------------

model = load_model("../vgg.h5")
# print(model.metrics_names)
model.summary()
# print('Data: TRAIN')
# print(model.evaluate(data_train, verbose=2))

# print('Data: TEST')
# print(print(model.evaluate(data_test, verbose=2)))

# print('Data: TEST_SIMPLE')
# print(model.evaluate(data_test, verbose=2))
# ---------------------------------------------
