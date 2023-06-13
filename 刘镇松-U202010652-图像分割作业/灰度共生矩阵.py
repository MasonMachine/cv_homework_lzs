import cv2
import numpy as np
from matplotlib import pyplot as plt
# 读取图像并将其转换为灰度图像

image_1 = cv2.imread('input_img/wood2.jpg',cv2.IMREAD_GRAYSCALE)
image_2 = cv2.imread('input_img/glcm.jpg',cv2.IMREAD_GRAYSCALE)

def glcm_get(img):
    # 获取图像尺寸
    glcm = np.zeros((256, 256), dtype=np.uint32)
    rows, cols = img.shape
    # 计算灰度共生矩阵glcm a=1 b=1
    for i in range(rows - 1):
        for j in range(cols - 1):
            glcm[img[i, j], img[i+1, j+1]] += 1

    return glcm

img_glcm_1 = glcm_get(image_1)
img_glcm_2 = glcm_get(image_2)
plt.subplot(221),plt.imshow(image_1,'gray'),plt.title("img1_original")
plt.subplot(222),plt.imshow(img_glcm_1,'gray'),plt.title("img1_glcm_a=1_b=1")
plt.subplot(223),plt.imshow(image_2,'gray'),plt.title("img2_original")
plt.subplot(224),plt.imshow(img_glcm_2,'gray'),plt.title("im2_glcm_a=1_b=1")
plt.tight_layout()
plt.savefig('output_img/image_GLCM_a1b1.jpg')
plt.show()