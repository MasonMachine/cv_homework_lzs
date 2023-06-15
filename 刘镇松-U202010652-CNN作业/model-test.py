# 手写数字识别 加载模型
from model import *
import cv2
import torch
import torchvision
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 加载模型
cnn_model = torch.load('data/cnn_model.pt',map_location='cpu')
cnn_model.eval()
print(cnn_model)
# 加载测试图片
image_path = "test_img/4.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, dsize=(28, 28), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
#print(img.shape)
plt.imshow(img, 'gray'), plt.title('ORIGINAL')
plt.show()
#输入测试图片img
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
img = transform(img)
img = torch.reshape(img,(1, 1, 28, 28))
#print(img.shape)
output = cnn_model(img)
_, prediction = torch.max(output.data, 1)
print(output)
print("prediction={}".format(prediction[0]))
