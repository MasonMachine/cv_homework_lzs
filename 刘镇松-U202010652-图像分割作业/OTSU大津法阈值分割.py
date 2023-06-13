import numpy as np
import matplotlib.pyplot as plt
import cv2

def threshold_processing(img, T):  # 阈值分割函数
    img1 = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= T:
                img1[i][j] = 1
            else:
                img1[i][j] = 0
    return img1

def otsu_threshold(image):
    pixel_num = image.shape[0] * image.shape[1]
    threshold = 0
    max_sigama_b = 0
    for k in range(256):
        P1 = np.sum(image <= k) / pixel_num  # 当阈值取k时，c1,c2类发生的概率P1,P2
        P2 = 1 - P1
        if P1 == 0 or P2 == 0:  # 如果c1,c2像素数为0，则无法计算方差，直接跳过该阈值
            continue
        m1 = np.mean(image[image <= k])  # 计算平均灰度m1,m2
        m2 = np.mean(image[image > k])
        sigma_b = P1 * P2 * ((m1 - m2) ** 2)
        if sigma_b > max_sigama_b:
            max_sigama_b = sigma_b
            threshold = k
    return threshold


img1 = cv2.imread('input_img/otsu.jpg', cv2.IMREAD_GRAYSCALE)  # 读取图像
t = otsu_threshold(img1)  # otsu算法得到阈值t1
img_otsu = threshold_processing(img1, t)  # 根据t进行阈值分割
# 展示处理结果
print('OTSU阈值t=', t)
plt.subplot(121), plt.imshow(img1, 'gray'), plt.title('ORIGINAL')
plt.subplot(122), plt.imshow(img_otsu, 'gray'), plt.title('OTSU')
plt.tight_layout()
plt.savefig('output_img/image_OTSU.jpg')
plt.show()

