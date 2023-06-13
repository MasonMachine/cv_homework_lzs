import numpy as np
import matplotlib.pyplot as plt
import cv2

# 定义高斯滤波函数
def gaussian_filter(img, kernel_size=3, sigma=1):
    """高斯滤波"""
    # 生成高斯卷积核
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))  # 生成高斯卷积核
    kernel = kernel / np.sum(kernel)  # 归一化

    # 对图像进行滤波
    filtered_img = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[2]):
        padded_img = np.pad(img[:, :, i], kernel_size // 2, mode='reflect')  # 边缘填充
        for x in range(filtered_img.shape[0]):
            for y in range(filtered_img.shape[1]):
                filtered_img[x, y, i] = np.sum(padded_img[x:x + kernel_size, y:y + kernel_size] * kernel)  # 卷积计算
    return filtered_img

#读取图像
img1=cv2.imread('input_img/LenaNoise.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#使用函数进行高斯滤波处理
img2 = gaussian_filter(img1, kernel_size=3, sigma=1)

#展示处理结果
#用plt展示
plt.subplot(121), plt.imshow(img1), plt.title('ORIGINAL')
plt.subplot(122), plt.imshow(img2), plt.title('GAUSSIAN')
plt.tight_layout()
plt.savefig('output_img/image_gaussian_filter.jpg')
plt.show()

# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# cv2.imwrite('output_img/image_gaussian.jpg', img2)
