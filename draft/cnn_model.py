# ------------------------------------LAYERS.EXP.PRE-----------------------------------
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
# ------------------------------------PREPROCESSING_IMG--------------------------------
from tensorflow.keras.preprocessing.image import img_to_array 
from tensorflow.keras.preprocessing.image import array_to_img 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
# ------------------------------------LAYERS-------------------------------------------
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dropout
# ------------------------------------OPTIMIZERS---------------------------------------
from tensorflow.keras.optimizers import RMSprop, Adam
# ------------------------------------LOSS---------------------------------------------
from tensorflow.keras.losses import CategoricalCrossentropy
# ------------------------------------MODELS-------------------------------------------
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
# ------------------------------------PIL----------------------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
model = Sequential([
    Rescaling(1./255, input_shape=(150, 150, 3)), 
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
model.compile(optimizer=RMSprop(learning_rate=0.0001, decay=1e-6), loss=CategoricalCrossentropy(), metrics=['accuracy'])
# Adam increase from epoch 13.
# model.compile(optimizer=Adam(learning_rate=0.0001, epsilon=1e-6), loss=CategoricalCrossentropy(), metrics=['accuracy'])
# ---------------------------------------------------------------------------------------
# model = load_model("main.h5")
 
