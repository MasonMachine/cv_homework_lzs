import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread("input_img/wood.jpg")
# 将图像转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 计算图像的傅里叶变换
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))
# 二值化处理傅里叶频谱图
normalized_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
_, binary_spectrum = cv2.threshold(normalized_spectrum, 125, 255, cv2.THRESH_BINARY)

# 可视化傅里叶频谱图
plt.subplot(131), plt.imshow(gray, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(binary_spectrum, cmap='gray')
plt.title('Magnitude Spectrum '), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig('output_img/image_Fourier_texture.jpg')
plt.show()
