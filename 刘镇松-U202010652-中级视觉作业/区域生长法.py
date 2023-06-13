import cv2
import numpy as np
import matplotlib.pyplot as plt


def fill_grow(img, seeds, threshold):
    height, width = img.shape
    label = np.zeros((height, width))
    output = np.zeros((height, width))
    label_flag = 0
    T = threshold
    seed_size = len(seeds)
    stack = []

    for i_seed in range(seed_size):
        stack.append(seeds[i_seed])
        label_flag = label_flag + 1
        total_size = 0
        avg_gray = img[seeds[i_seed][0], seeds[i_seed][1]]
        while stack != []:
            a = stack.pop()
            if label[a[0], a[1]] == 0:
                total_size = total_size + 1
                label[a[0], a[1]] = 1
                avg_gray = (avg_gray * (total_size - 1) + img[a[0], a[1]]) / total_size
                for i in range(8):
                    tmpX = a[0] + connects[i].x
                    tmpY = a[1] + connects[i].y
                    if (tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= width):
                        continue
                    gray_new = img[tmpX, tmpY]
                    dist = np.sqrt(np.sum(np.square(avg_gray - gray_new)))
                    if dist < T:
                        output[tmpX, tmpY] = label_flag
                        stack.append([tmpX, tmpY])
    return output

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0),
            Point(1, 1), Point(0, 1), Point(-1, 1), Point(-1, 0)]


img1 = cv2.imread('input_img/area_grow.jpg',cv2.IMREAD_GRAYSCALE)
seeds=[[100,200],[200,100],[100,100],[200,200],[250,250]]
img_res=fill_grow(img1,seeds,30)

plt.subplot(121), plt.imshow(img1,'gray'), plt.title('ori')
plt.subplot(122), plt.imshow(img_res,'gray'), plt.title('seed_fill')
plt.tight_layout()
plt.savefig('output_img/image_seed_fill.jpg')
plt.show()
