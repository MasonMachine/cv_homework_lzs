import numpy as np
import matplotlib.pyplot as plt
import cv2

#定义中值滤波函数
def median_filter(img, kernel_size=3):
    # 对图像进行滤波
    filtered_img = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[2]):
        padded_img = np.pad(img[:, :, i], kernel_size // 2, mode='reflect')
        for x in range(filtered_img.shape[0]):
            for y in range(filtered_img.shape[1]):
                filtered_img[x, y, i] = np.median(padded_img[x:x+kernel_size, y:y+kernel_size])
    return filtered_img

#读取图像
img1=cv2.imread('input_img/LenaNoise.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

#使用函数进行中值滤波处理
img2 = median_filter(img1)

#展示处理结果
plt.subplot(121), plt.imshow(img1), plt.title('ORIGINAL')
plt.subplot(122), plt.imshow(img2), plt.title('MIDDLE')
plt.tight_layout()
plt.savefig('output_img/image_middle_filter.jpg')
plt.show()

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
cv2.imwrite("output_img/image_middle.jpg",img2)
