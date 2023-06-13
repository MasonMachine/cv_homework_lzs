import numpy as np
import matplotlib.pyplot as plt
import cv2

# 定义膨胀函数
def image_dilation(image, kernel):
    # 获取图像的尺寸和核的尺寸
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # 计算膨胀操作的填充量
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    # 创建用于存储膨胀结果的新图像
    dilated_image = np.zeros_like(image)

    # 边缘填充
    padded_img = np.pad(image, (padding_height, padding_width), mode='reflect')

    # 对图像进行膨胀操作
    for i in range(image_height):
        for j in range(image_width):
            # 将核放置在当前像素位置
            image_patch = padded_img[i:i + kernel_height, j:j + kernel_width]

            # 使用逐元素的最大值操作来实现膨胀
            dilated_value = np.max(image_patch * kernel)

            # 将膨胀后的像素值赋给新图像
            dilated_image[i, j] = dilated_value

    return dilated_image


# 定义腐蚀函数
def image_erosion(image, kernel):
    # 获取图像的尺寸和核的尺寸
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # 计算腐蚀操作的填充量
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    # 创建用于存储腐蚀结果的新图像
    eroded_image = np.zeros_like(image)

    # 边缘填充
    padded_img = np.pad(image, (padding_height, padding_width), mode='reflect')

    # 对图像进行腐蚀操作
    for i in range(image_height):
        for j in range(image_width):
            # 将核放置在当前像素位置
            image_patch = padded_img[i:i + kernel_height, j:j + kernel_width]

            # 使用逐元素的最小值操作来实现腐蚀
            eroded_value = np.min(image_patch * kernel)

            # 将腐蚀后的像素值赋给新图像
            eroded_image[i, j] = eroded_value

    return eroded_image


img1 = cv2.imread('input_img/cube.jpg', cv2.IMREAD_GRAYSCALE)  # 读取图像
kernel = np.ones((3, 3), np.uint8)
img_dilation = image_dilation(img1, kernel)
img_erosion = image_erosion(img1, kernel)

# 展示膨胀腐蚀处理结果
plt.subplot(131), plt.imshow(img1, 'gray'), plt.title('ORIGINAL')
plt.subplot(132), plt.imshow(img_dilation, 'gray'), plt.title('DILATION')
plt.subplot(133), plt.imshow(img_erosion, 'gray'), plt.title('EROSION')
plt.tight_layout()
plt.savefig('output_img/image_dilation_erosion.jpg')
plt.show()
