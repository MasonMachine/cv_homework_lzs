import numpy as np
import matplotlib.pyplot as plt
import cv2

def Origin_histogram(img):
    # 建立原始图像各灰度级的灰度值与像素个数对应表
    histogram = {}
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            k = img[i][j]
            if k in histogram:
                histogram[k] += 1
            else:
                histogram[k] = 1

    sorted_histogram = {}  # 建立排好序的映射表（字典）
    sorted_list = sorted(histogram)  # 根据灰度值进行从低至高的排序，得到按键从小到大排序的键值对

    for j in range(len(sorted_list)):
        sorted_histogram[sorted_list[j]] = histogram[sorted_list[j]]

    return sorted_histogram


def equalization_histogram(histogram, img):
    pr = {}  # 建立概率分布映射表（字典）

    for i in histogram.keys():
        pr[i] = histogram[i] / (img.shape[0] * img.shape[1])  # 计算对应灰度值的频率

    tmp = 0
    for m in pr.keys():
        tmp += pr[m]
        pr[m] = max(histogram) * tmp  # 对应的值为Dm*频率累加值

    new_img = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)

    for k in range(img.shape[0]):
        for l in range(img.shape[1]):
            new_img[k][l] = pr[img[k][l]]

    return new_img


img1=cv2.imread('input_img/ball.jpg',cv2.IMREAD_GRAYSCALE) #读取图像
origin_histogram = Origin_histogram(img1)
#直方图均衡化
equ = equalization_histogram( origin_histogram, img1 )

#展示处理结果
plt.subplot(221), plt.hist(img1.ravel(),256), plt.title('ORIGINAL_HIST')
plt.subplot(222), plt.hist(equ.ravel(),256), plt.title('EQUALIZE_HIST')
plt.subplot(223), plt.imshow(img1,'gray'), plt.title('ORIGINAL_HIST')
plt.subplot(224), plt.imshow(equ,'gray'), plt.title('EQUALIZE_HIST')
plt.tight_layout()
plt.savefig('output_img/image_hist_equ.jpg')
plt.show()

# cv2.imwrite('output_img/image_hist_equ.jpg', equ)

