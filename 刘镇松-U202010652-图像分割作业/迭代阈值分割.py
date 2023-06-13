import numpy as np
import matplotlib.pyplot as plt
import cv2

# 迭代阈值处理函数
def threshold_processing(img, T):  # 阈值分割函数
    img1 = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= T:
                img1[i][j] = 1
            else:
                img1[i][j] = 0
    return img1


def Iterative_threshold(img, T_th=0.0001, max_iterations=100,
                        initial_threshold=128):  # 迭代算法得到阈值T,给的参数有最大迭代次数和迭代误差，以及初始阈值
    T = initial_threshold
    T_new = 0
    for k in range(max_iterations):
        g1 = img[img >= T]  # 将图片分类为两个数组
        g2 = img[img < T]
        m1 = np.mean(g1)
        m2 = np.mean(g2)
        T_new = (m1 + m2) * 0.5
        if np.abs(T_new - T) < T_th:  # 迭代终止条件
            break
        T = T_new

    return T


img1 = cv2.imread('input_img/finger_print.jpg', cv2.IMREAD_GRAYSCALE)  # 读取图
t = Iterative_threshold(img1)  # 迭代算法得到阈值t
img_iter = threshold_processing(img1, t)  # 根据t进行阈值分割

# 展示处理结果
print('迭代阈值t=', t)
plt.subplot(121), plt.imshow(img1, 'gray'), plt.title('ORIGINAL')
plt.subplot(122), plt.imshow(img_iter, 'gray'), plt.title('ITERATIVE_THRESHOLD')
plt.tight_layout()
plt.savefig('output_img/image_Iterative_threshold.jpg')
plt.show()
