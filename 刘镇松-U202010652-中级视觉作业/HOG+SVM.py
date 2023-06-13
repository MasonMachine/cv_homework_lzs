import cv2 as cv
from matplotlib import pyplot as plt


image = cv.imread("input_img/walking.jpg")
image_ori = image.copy()
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
# Detect people in the image
(rects, weights) = hog.detectMultiScale(image,winStride=(4, 4),padding=(8, 8),scale=1.25,useMeanshiftGrouping=False)
for (x, y, w, h) in rects:
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image_ori = cv.cvtColor(image_ori,cv.COLOR_BGR2RGB)
plt.subplot(121), plt.imshow(image_ori), plt.title('ORIGINAL')
plt.subplot(122), plt.imshow(image), plt.title('HOG+SVM')
plt.tight_layout()
plt.savefig('output_img/image_HOG+SVM.jpg')
plt.show()