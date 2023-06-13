import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

# 定义在x,y处的自相关函数
def image_autocorrelation(image, x=0, y=0):
    image_height, image_width = image.shape
    image = image / 255.0
    temp = 0
    f1 = 0
    sum_f = 0
    for i in range(image_height):
        for j in range(image_width):
            if (i + x) >= image_height or (j + y) >= image_width:
                f1 = 0
            else:
                f1 = image[i + x, j + y]
            temp = temp + image[i, j] * f1

    sum_f = np.sum(image * image)  # 求平方和
    autocorrelation = temp / sum_f  # 求自相关函数f(x,y)
    return autocorrelation

img1 = cv2.imread('input_img/wood2.jpg',cv2.IMREAD_GRAYSCALE) #读取图像
h = 10
w = 10
z =np.zeros((h,w))
for x in range(h):
    for y in range(w):
         z[x,y] = image_autocorrelation(img1, x*10, y*10)


# 创建数据
x = range(0,100,10)
y = range(0,100,10)
# 绘制线图
plt.plot(x, z[0,:])
# 添加标题和轴标签
plt.title("Autocorrelation in X")
plt.xlabel("X")
plt.ylabel("Autocorrelation")
plt.savefig('output_img/image_autocorrelation1.jpg')
plt.show()
# 绘制线图
plt.plot(y, z[0,:])
# 添加标题和轴标签
plt.title("Autocorrelation in Y")
plt.xlabel("Y")
plt.ylabel("Autocorrelation")
# 显示图形
plt.savefig('output_img/image_autocorrelation2.jpg')
plt.show()

# 3d图像
# 创建数据
x = range(0,100,10)
y = range(0,100,10)
X, Y = np.meshgrid(x, y)
# 创建 3D 图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 绘制三维图形
ax.plot_surface(X, Y, z, cmap='viridis')
# 添加标题和轴标签
ax.set_title('Autocorrelation in X and Y')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('output_img/image_autocorrelation3.jpg')
# 显示图形
plt.show()
