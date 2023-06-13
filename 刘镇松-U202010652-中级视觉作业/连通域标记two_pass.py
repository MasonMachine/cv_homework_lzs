import cv2
import numpy as np
import matplotlib.pyplot as plt

#定义迭代阈值处理函数，用于将rgb转为二值图
def threshold_processing(img , T):#阈值分割函数
    img1 = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >=T :
                img1[i][j]=255
            else :
                img1[i][j]=0
    return img1

def Iterative_threshold(img ,T_th=0.0001 ,max_iterations=100 ,initial_threshold=128): #迭代算法得到阈值T,给的参数有最大迭代次数和迭代误差，以及初始阈值
    T = initial_threshold
    T_new = 0
    for k in range(max_iterations):
        g1 = img[img >= T] #将图片分类为两个数组
        g2 = img[img < T]
        m1 = np.mean(g1)
        m2 = np.mean(g2)
        T_new = (m1+m2)*0.5
        if np.abs(T_new-T)<T_th: #迭代终止条件
            break
        T =T_new
    return T

def two_pass(img):
    T = Iterative_threshold(img)
    img_1 = threshold_processing(img, T)
    img_1 = abs(img_1 - 255)
    h, w = img.shape
    label = np.zeros((h, w))
    img_output = np.zeros((h, w))
    label_flag = 1

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if img_1[i, j] != 0:
                if (label[i - 1, j] == 0) and (label[i, j - 1] == 0):
                    label[i, j] = label_flag
                    label_flag = label_flag + 1
                elif (label[i - 1, j] == 0) and (label[i, j - 1] != 0):
                    label[i, j] = label[i, j - 1]
                elif (label[i - 1, j] != 0) and (label[i, j - 1] == 0):
                    label[i, j] = label[i - 1, j]
                else:
                    label[i, j] = min(label[i - 1, j], label[i, j - 1])
    stack = []

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if label[i, j] != 0 and img_output[i, j] == 0:
                stack.append([i, j])
                while stack != []:
                    a = stack.pop()
                    if label[a[0], a[1]] != 0:
                        if (img_output[a[0] - 1, a[1]] == 0 and label[a[0] - 1, a[1]] != 0): stack.append(
                            [a[0] - 1, a[1]])
                        if (img_output[a[0] + 1, a[1]] == 0 and label[a[0] + 1, a[1]] != 0): stack.append(
                            [a[0] + 1, a[1]])
                        if (img_output[a[0], a[1] - 1] == 0 and label[a[0], a[1] - 1] != 0): stack.append(
                            [a[0], a[1] - 1])
                        if (img_output[a[0], a[1] + 1] == 0 and label[a[0], a[1] + 1] != 0): stack.append(
                            [a[0], a[1] + 1])
                        img_output[a[0], a[1]] = label[i, j]

    return img_output

img1=cv2.imread('input_img/blocks.jpg',cv2.IMREAD_GRAYSCALE) #读取图像
img_two_pass=two_pass(img1)
plt.subplot(121), plt.imshow(img1,'gray'), plt.title('ORIGINAL')
plt.subplot(122), plt.imshow(img_two_pass,'gray'), plt.title('ORIGINAL')
plt.tight_layout()
plt.savefig('output_img/image_twopass.jpg')
plt.show()