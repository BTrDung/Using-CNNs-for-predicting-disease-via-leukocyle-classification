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
  
xTest = [] 
yTest = [] 
import os 

arr = os.listdir('E:\\SelfStudyTryHard\\CS213\\dataset2-master\\images\\TEST\\EOSINOPHIL')
for link in arr: 
    xTest.append([getImg('E:\\SelfStudyTryHard\\CS213\\dataset2-master\\images\\TEST\\EOSINOPHIL\\' + link)])
    yTest.append([0])


arr = os.listdir('E:\\SelfStudyTryHard\\CS213\\dataset2-master\\images\\TEST\\LYMPHOCYTE')
for link in arr: 
    xTest.append([getImg('E:\\SelfStudyTryHard\\CS213\\dataset2-master\\images\\TEST\\LYMPHOCYTE\\' + link)])
    yTest.append([1])


arr = os.listdir('E:\\SelfStudyTryHard\\CS213\\dataset2-master\\images\\TEST\\MONOCYTE\\')
for link in arr: 
    xTest.append([getImg('E:\\SelfStudyTryHard\\CS213\\dataset2-master\\images\\TEST\\MONOCYTE\\' + link)])
    yTest.append([2])


arr = os.listdir('E:\\SelfStudyTryHard\\CS213\\dataset2-master\\images\\TEST\\NEUTROPHIL')
for link in arr: 
    xTest.append([getImg('E:\\SelfStudyTryHard\\CS213\\dataset2-master\\images\\TEST\\NEUTROPHIL\\' + link)])
    yTest.append([3])


xTest = np.array(xTest)  
xTest = xTest.reshape(len(xTest), 240, 320, 3)
yTest = np.array(yTest)
