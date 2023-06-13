import numpy as np
import cv2

# harris角点检测
def harris_corner_detection(image, threshold=0.01, window_size=3, k=0.04):
    # 计算图像的梯度
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度的乘积
    dx2 = dx * dx
    dy2 = dy * dy
    dxy = dx * dy

    # 使用窗口滤波器对梯度乘积进行平滑
    kernel = np.ones((window_size, window_size), dtype=np.float32)
    dx2 = cv2.filter2D(dx2, -1, kernel)
    dy2 = cv2.filter2D(dy2, -1, kernel)
    dxy = cv2.filter2D(dxy, -1, kernel)

    # 计算Harris响应函数
    det = dx2 * dy2 - dxy * dxy
    trace = dx2 + dy2
    harris_response = det - k * trace * trace  # 得到R

    # 通过阈值处理找到角点
    corners = np.zeros_like(image)
    corners[harris_response > threshold * harris_response.max()] = 255  # 选择R大于阈值的且为

    return corners


img1 = cv2.imread('input_img/point.jpg',cv2.IMREAD_GRAYSCALE) #读取图像
# 进行Harris角点检测
corners = harris_corner_detection(img1,0.2)

# 在原始图像上绘制角点
img2 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
image_with_corners = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
image_with_corners[corners != 0] = [0, 0, 255]

# 显示原始图像和带有角点的图像
cv2.imshow('Original Image', img2)
cv2.imshow('Image with Corners', image_with_corners)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('output_img/image_Harris.jpg',image_with_corners)