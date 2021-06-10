import collections
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
 
from PIL import Image

def getImg(link): 
    with open(link, 'rb') as f: 
        img = Image.open(f)
        img = img.convert('RGB')
        # img.show()
        return np.array(img)
    return None 

img     = [ [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\EOSINOPHIL\_0_5239.jpeg')], 
            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\EOSINOPHIL\_1_5031.jpeg')],
            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\EOSINOPHIL\_2_1226.jpeg')],
            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\EOSINOPHIL\_3_625.jpeg')],
            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\EOSINOPHIL\_4_8814.jpeg')],
            
            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\LYMPHOCYTE\_0_3975.jpeg')],
            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\LYMPHOCYTE\_1_4044.jpeg')],
            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\LYMPHOCYTE\_2_6981.jpeg')],
            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\LYMPHOCYTE\_3_7545.jpeg')],
            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\LYMPHOCYTE\_4_2908.jpeg')],

            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\MONOCYTE\_0_5020.jpeg')],
            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\MONOCYTE\_1_4511.jpeg')],
            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\MONOCYTE\_2_4392.jpeg')],
            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\MONOCYTE\_3_9457.jpeg')],
            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\MONOCYTE\_3_9457.jpeg')],

            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\\NEUTROPHIL\_0_1966.jpeg')],
            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\\NEUTROPHIL\_1_2118.jpeg')],
            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\\NEUTROPHIL\_2_1918.jpeg')],
            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\\NEUTROPHIL\_3_1003.jpeg')],
            [getImg('E:\SelfStudyTryHard\CS213\dataset2-master\images\TEST_SIMPLE\\NEUTROPHIL\_4_1395.jpeg')]
        ]

img     = np.array(img)
img     = img.reshape(20, 240, 320, 3)
xTrain  = img 
yTrain = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [2], [2], [2], [2], [2], [3], [3], [3], [3], [3]])
 
