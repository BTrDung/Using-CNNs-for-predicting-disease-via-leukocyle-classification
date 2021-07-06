import os
import time
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# -------------------SOURCE-IMAGE---------------------------------
result = []

# def locate_center(ori_y, ori_x):
#     x, y = 0, 0
#     if ori_x >= 112 and ori_x <= 720 -112:
#         x = ori_x
#     else:
#         if ori_x <112:
#             x = 112
#         if ori_x > 720 - 112:
#             x = 720 - 112
#     if ori_y >= 112 and ori_y <= 576 - 112:
#         y = ori_y
#     else:
#         if ori_y < 112:
#             y = 112
#         if ori_y > 576 - 112:
#             y = 576 - 112
#     return y, x

def convert(path_img, path_msk):
    img = cv.imread(path_img)
    img_rgb = np.zeros((img.shape[0] + 224, img.shape[1] + 224, 3))
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            img_rgb[i + 112][j + 112] = img[i][j]

    img = cv.imread(path_msk)
    img = cv.resize(img, (720, 576), interpolation=cv.INTER_AREA)
    img_msk = np.zeros((img.shape[0] + 224, img.shape[1] + 224, 3))
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            img_msk[i + 112][j + 112] = img[i][j]

    mask = np.copy(img_msk)
    mask[img_msk == 0] = 0
    mask[img_msk != 0] = 255
    cv.imwrite('original_data/mask.png', mask)
    mask = cv.imread('original_data/mask.png')
    img_gray = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
    img_canny = cv.Canny(mask, 255, 255)

    circles = cv.HoughCircles(img_canny, cv.HOUGH_GRADIENT, 1, img_canny.shape[0] / 15, param1=200, param2=10,
                              minRadius=20, maxRadius=50)
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0, :]:
    #         center = (i[0], i[1])
    #         cv.circle(img_rgb, center, 1, (0, 100, 100), 3)
    #         radius = i[2]
    #         # cv.circle(img_rgb, center, radius, (255, 0, 255), 3)
    #
    # cv.imshow('vcl', img_rgb)
    # # cv.imwrite('output.png', img_rgb)
    #
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # for i in circles[0, :]:
        for i in circles[0, :]:
            center = (i[0], i[1])
            # y, x = locate_center(i[0], i[1])
            crop = np.copy(img_rgb[(i[1] - 112):(i[1] + 112), (i[0] - 112):(i[0] + 112)])
            # crop = np.copy(img_rgb[(y - 112):(y + 112), (x - 112):(x + 112)])
            result.append(crop)

# -------------------BAOSO---------------------------------
for i in range(1, 54):
    print(i)
    path_img = 'LISC_Database/Main Dataset/Baso/' + str(i) + '.bmp'
    path_msk = 'LISC_Database/Ground Truth Segmentation/Baso/areaforexpert1/' + str(i) + '_expert.bmp'
    convert(path_img, path_msk)

for i in range(0, len(result)):
    cv.imwrite('original_data/Baoso/' + str(i) + '.png', result[i])


# -------------------EOSI---------------------------------
# for i in range(1, 39):
#     print(i)
#     path_img = 'LISC_Database/Main Dataset/eosi/' + str(i) + '.bmp'
#     path_msk = 'LISC_Database/Ground Truth Segmentation/eosi/areaforexpert1/' + str(i) + '_expert.bmp'
#     convert(path_img, path_msk)
#
# for i in range(0, len(result)):
#     cv.imwrite('original_data/eosi/' + str(i) + '.png', result[i])

# -------------------LYMP---------------------------------
# for i in range(1, 52):
#     print(i)
#     path_img = 'LISC_Database/Main Dataset/lymp/' + str(i) + '.bmp'
#     path_msk = 'LISC_Database/Ground Truth Segmentation/lymp/areaforexpert1/' + str(i) + '_expert.bmp'
#     convert(path_img, path_msk)
#
# for i in range(0, len(result)):
#     cv.imwrite('original_data/lymp/' + str(i) + '.png', result[i])
# i = 33
# path_img = 'E:/Github/Machine-learning-for-counting-blood-cells/data2/Main Dataset/lymp/' + str(i) + '.bmp'
# path_msk = 'E:/Github/Machine-learning-for-counting-blood-cells/data2/Ground Truth Segmentation/lymp/areaforexpert1/' + str(i) + '_expert.bmp'
# convert(path_img, path_msk)


# -------------------MIXT---------------------------------
# for i in range(1, 8):
#     print(i)
#     path_img = 'E:/Github/Machine-learning-for-counting-blood-cells/data2/Main Dataset/mixt/' + str(i) + '.bmp'
#     path_msk = 'E:/Github/Machine-learning-for-counting-blood-cells/data2/Ground Truth Segmentation/mixt/areaforexpert1/' + str(i) + '_expert.bmp'
#     convert(path_img, path_msk)

# for i in range(0, len(result)):
#     cv.imwrite('E:/Github/Machine-learning-for-counting-blood-cells/data2/mixt/' + str(i) + '.png', result[i])


# -------------------MONO---------------------------------
# for i in range(1, 48):
#     print(i)
#     path_img = 'LISC_Database/Main Dataset/mono/' + str(i) + '.bmp'
#     path_msk = 'LISC_Database/Ground Truth Segmentation/mono/areaforexpert1/' + str(i) + '_expert.bmp'
#     convert(path_img, path_msk)
#
# for i in range(0, len(result)):
#     cv.imwrite('original_data/mono/' + str(i) + '.png', result[i])


# -------------------NEUT---------------------------------
# for i in range(47, 48):
#     print(i)
#     path_img = 'LISC_Database/Main Dataset/neut/' + str(i) + '.bmp'
#     path_msk = 'LISC_Database/Ground Truth Segmentation/neut/areaforexpert1/' + str(i) + '_expert.bmp'
#     convert(path_img, path_msk)
#
# for i in range(0, len(result)):
#     cv.imwrite('original_data/neut/' + str(i) + '.png', result[i])

