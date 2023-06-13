import numpy as np
import cv2
import matplotlib.pyplot as plt

def GLPF(image, d0=50):  # 二阶高斯低通滤波器
    H = np.zeros_like(image, float)
    M, N = image.shape
    mid_x = M / 2
    mid_y = N / 2
    # 传递函数计算
    for x in range(0, M):
        for y in range(0, N):
            d = np.sqrt((x - mid_x) ** 2 + (y - mid_y) ** 2)
            H[x, y] = np.exp(-d ** 2 / (2 * d0 ** 2))

    fftImg = np.fft.fft2(image)  # 对图像进行傅里叶变换
    fftImgShift = np.fft.fftshift(fftImg)  # 傅里叶变换后坐标移动到图像中心
    handle_fftImgShift1 = fftImgShift * H  # 对傅里叶变换后的图像进行频域变换，乘传递函数
    handle_fftImgShift2 = np.fft.ifftshift(handle_fftImgShift1)  # 反移动
    handle_fftImgShift3 = np.fft.ifft2(handle_fftImgShift2)  # 傅里叶反变换
    handle_fftImgShift4 = np.real(handle_fftImgShift3)  # 傅里叶反变换后取实部
    return np.uint8(handle_fftImgShift4)

img1=cv2.imread('input_img/letter.jpg',cv2.IMREAD_GRAYSCALE)#读取灰度图
#进行高斯低通滤波运算，设置不同的参数
img2=GLPF(img1,10)
img3=GLPF(img1,30)
img4=GLPF(img1,60)
#高斯低通滤波结果展示
#plt.show()展示
plt.subplot(221), plt.imshow(img1,'gray'), plt.title('ORIGINAL')
plt.subplot(222), plt.imshow(img2,'gray'), plt.title('GLPF_D0=10')
plt.subplot(223), plt.imshow(img3,'gray'), plt.title('GLPF_D0=30')
plt.subplot(224), plt.imshow(img4,'gray'), plt.title('GLPF_D0=60')
plt.tight_layout()
plt.savefig('output_img/image_GLPF.jpg')
plt.show()

