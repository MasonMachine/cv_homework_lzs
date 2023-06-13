import numpy as np
import cv2
import matplotlib.pyplot as plt

# 局部共生描述LBP
def LBP(src):
    height = src.shape[0]
    width = src.shape[1]
    dst = src.copy()

    lbp_value = np.zeros((1, 8), dtype=np.uint8)
    neighbours = np.zeros((1, 8), dtype=np.uint8)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]
            center = src[y, x]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 + lbp_value[
                0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 7] * 128
            dst[y, x] = lbp.item()
    return dst


image = cv2.imread('input_img/label.jpg')
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
LBP_img = LBP(img_gray)

plt.subplot(121), plt.imshow(img_gray, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(LBP_img, cmap='gray'), plt.title('LBP Image')
plt.tight_layout()
plt.savefig('output_img/image_LBP.jpg')
plt.show()
