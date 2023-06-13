import numpy as np
import matplotlib.pyplot as plt
import cv2

# 定义伽马变换函数 参数为c和γ c默认为1
def gamma_trans(image, gamma, c=1):
    image = image / 255.0  # 归一化
    temp = c * np.power(image, gamma)  # 幂次运算
    temp = temp * 255  # 转化为8bit类型
    #     防止溢出
    [rows, cols] = temp.shape
    for i in range(rows):
        for j in range(cols):
            if temp[i, j] > 255:
                temp[i, j] = 255
    New = temp.astype(int)  # 取整

    return New

img1=cv2.imread('input_img/ball.jpg',cv2.IMREAD_GRAYSCALE) #读取图像
img2=gamma_trans(img1,0.4,c=1.1)#进行伽马变换,gamma=0.4,c=1.1
img3=gamma_trans(img1,1)#gamma=1
img4=gamma_trans(img1,2.5)#gamma=2.5

#展示处理结果
plt.subplot(221), plt.imshow(img1,'gray'), plt.title('ORIGINAL')
plt.subplot(222), plt.imshow(img2,'gray'), plt.title('GAMMA=0.4')
plt.subplot(223), plt.imshow(img3,'gray'), plt.title('GAMMA=1')
plt.subplot(224), plt.imshow(img4,'gray'), plt.title('GAMMA=2.5')
plt.tight_layout()
plt.savefig('output_img/image_gamma.jpg')
plt.show()

# cv2.imwrite('output_img/image_gamma_0.4.jpg', img2)
# cv2.imwrite('output_img/image_gamma_1.jpg', img3)
# cv2.imwrite('output_img/image_gamma_2.5.jpg', img4)
