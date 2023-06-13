import numpy as np
import matplotlib.pyplot as plt
import cv2
from 膨胀与腐蚀 import *

# 开运算=先腐蚀再膨胀
def image_opening(image, kernel):
    # 执行腐蚀操作
    eroded_image = image_erosion(image, kernel)
    # 执行膨胀操作
    opened_image = image_dilation(eroded_image, kernel)
    return opened_image

# 闭运算=先膨胀再腐蚀
def image_closing(image, kernel):
    # 执行膨胀操作
    dilated_image = image_dilation(image, kernel)
    # 执行腐蚀操作
    closed_image = image_erosion(dilated_image, kernel)
    return closed_image


img1 = cv2.imread('input_img/cube.jpg',cv2.IMREAD_GRAYSCALE) # 读取图像
# 生成对应的核并进行运算
kernel = np.ones((3,3),np.uint8)
img_opening = image_opening(img1 , kernel)
img_closing = image_closing(img1 , kernel)
#展示开闭运算处理结果以及膨胀腐蚀运算结果
plt.subplot(231), plt.imshow(img1,'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(img_opening,'gray'), plt.title('OPENING')
plt.subplot(233), plt.imshow(img_closing,'gray'), plt.title('CLOSING')
plt.subplot(234), plt.imshow(img1,'gray'), plt.title('ORIGINAL')
plt.subplot(235), plt.imshow(img_dilation,'gray'), plt.title('DILATION')
plt.subplot(236), plt.imshow(img_erosion,'gray'), plt.title('EROSION')
plt.tight_layout()
plt.savefig('output_img/image_open_close.jpg')
plt.show()
