import collections
import numpy as np 
 
from PIL import Image

def getImg(link): 
    with open(link, 'rb') as f: 
        img = Image.open(f)
        img = img.convert('RGB')
        # img.show()
        return np.array(img)
    return None 
  
xTrain = [] 
yTrain = [] 
import os 

arr = os.listdir('E:\\SelfStudyTryHard\\CS213\\dataset2-master\\images\\TRAIN_SIMPLE\\EOSINOPHIL')
for link in arr: 
    xTrain.append([getImg('E:\\SelfStudyTryHard\\CS213\\dataset2-master\\images\\TRAIN_SIMPLE\\EOSINOPHIL\\' + link)])
    yTrain.append([0])


arr = os.listdir('E:\\SelfStudyTryHard\\CS213\\dataset2-master\\images\\TRAIN_SIMPLE\\LYMPHOCYTE')
for link in arr: 
    xTrain.append([getImg('E:\\SelfStudyTryHard\\CS213\\dataset2-master\\images\\TRAIN_SIMPLE\\LYMPHOCYTE\\' + link)])
    yTrain.append([1])


arr = os.listdir('E:\\SelfStudyTryHard\\CS213\\dataset2-master\\images\\TRAIN_SIMPLE\\MONOCYTE\\')
for link in arr: 
    xTrain.append([getImg('E:\\SelfStudyTryHard\\CS213\\dataset2-master\\images\\TRAIN_SIMPLE\\MONOCYTE\\' + link)])
    yTrain.append([2])


arr = os.listdir('E:\\SelfStudyTryHard\\CS213\\dataset2-master\\images\\TRAIN_SIMPLE\\NEUTROPHIL')
for link in arr: 
    xTrain.append([getImg('E:\\SelfStudyTryHard\\CS213\\dataset2-master\\images\\TRAIN_SIMPLE\\NEUTROPHIL\\' + link)])
    yTrain.append([3])


xTrain = np.array(xTrain)  
xTrain = xTrain.reshape(len(xTrain), 240, 320, 3)
yTrain = np.array(yTrain)
