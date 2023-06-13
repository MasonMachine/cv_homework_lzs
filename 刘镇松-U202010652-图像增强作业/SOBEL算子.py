import numpy as np
import cv2
import matplotlib.pyplot as plt

#定义sobel算子函数
def sobel_operator(img):
    padded_img= np.pad(img,3//2, mode='reflect')#填充
    r, c = img.shape
    new_image = np.zeros(img.shape)
    new_imageX = np.zeros(img.shape)
    new_imageY = np.zeros(img.shape)
    # X方向算子和Y方向算子
    s_operatorX = np.array([[-1,-2,-1],
                          [0,0,0],
                          [1,2,1]])
    s_operatorY = np.array([[-1,0,1],
                          [-2,0,2],
                          [-1,0,1]])
    for i in range(r):
        for j in range(c):
            new_imageX[i, j] = abs(np.sum(padded_img[i:i+3, j:j+3] * s_operatorX))
            new_imageY[i, j] = abs(np.sum(padded_img[i:i+3, j:j+3] * s_operatorY))
            new_image[i, j] = (new_imageX[i, j]*new_imageX[i,j] + new_imageY[i, j]*new_imageY[i,j])**0.5
    #return np.uint8(new_imageX) X方向算子处理
    #return np.uint8(new_imageY) Y方向算子处理
    return (np.uint8(new_image),np.uint8(new_imageX),np.uint8(new_imageY))  # 依次返回X方向, Y方向,无方向算子处理的图像

img1=cv2.imread('input_img/circle.jpg',cv2.IMREAD_GRAYSCALE) # 读取灰度图
(img2,img2_x,img2_y) = sobel_operator(img1) # 用sobel算子处理图像

#结果展示
#plt.show()展示处理的边缘图
plt.subplot(221), plt.imshow(img1,'gray'), plt.title('ORIGINAL')
plt.subplot(222), plt.imshow(img2,'gray'), plt.title('SOBEL_MUTI')
plt.subplot(223), plt.imshow(img2_x,'gray'), plt.title('SOBEL_X')
plt.subplot(224), plt.imshow(img2_y,'gray'), plt.title('SOBEL_Y')
plt.tight_layout()
plt.savefig('output_img/image_SOBEL.jpg')
plt.show()

# cv2.imwrite("output_img/image_sobel.jpg",img2)
# cv2.imwrite("output_img/image_sobel_x.jpg",img2_x)
# cv2.imwrite("output_img/image_sobel_y.jpg",img2_y)

