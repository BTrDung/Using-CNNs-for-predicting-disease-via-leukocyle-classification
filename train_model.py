from draw_plt_val_accur import draw_plt
from cnn_model import model
from init_data_train import data_train
import tensorflow
# from init_data_train import val_ds

# -------------------------------------------------------------------------------------
# cache if enough memory
# AUTOTUNE = tensorflow.data.AUTOTUNE
# data_train = data_train.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# model.summary()
epoch = 6
history = model.fit(data_train, batch_size=30, epochs=epoch, verbose=1)
# model.fit(data_train, validation_data=val_ds, batch_size=30, epochs=30, verbose=1)


model.save("main.h5")

# draw plt if use validation_data
# draw_plt(history, epoch)
