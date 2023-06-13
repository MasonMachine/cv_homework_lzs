import numpy as np
import cv2

cap = cv2.VideoCapture('input_img/meanshift.mp4')
# 获取视频的第一帧
ret, frame = cap.read()
# 设置初始窗口位置
x, y, w, h = 480, 300, 75,75 # 硬编码位置
track_window = (x, y, w, h)
# 对追踪对象设置ROI
roi = frame[y:y + h, x:x + w]
# 只考虑HSV的色调
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# 为了避免由于低光导致的错误值，使用 cv2.inRange() 函数丢弃低光值。
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
# 设置终止标准，10 次迭代或移动至少 1pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while (1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # 应用meanshift获取新位置
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # 在图像上绘制它
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv2.imshow('img2', img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k) + ".jpg", img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()
