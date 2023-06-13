import cv2
import numpy as np
from matplotlib import pyplot as plt

img2 = cv2.imread('input_img/Lena.jpeg')
img2_c = img2.copy()
img2 = cv2.GaussianBlur(img2, (3, 3), 0)
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
for line in lines:
    rho = line[0][0]
    theta = line[0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img2_c, (x1, y1), (x2, y2), (0, 0, 255), 2)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, 300, 10)

# 展示结果
img2_c = cv2.cvtColor(img2_c, cv2.COLOR_BGR2RGB)
plt.imshow(img2_c), plt.title('houghlines')
plt.savefig('output_img/image_hough_lines.jpg')
plt.show()