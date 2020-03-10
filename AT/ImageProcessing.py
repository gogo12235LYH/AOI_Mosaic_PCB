"""
Created on Sat Nov  9 23:25:43 2019

@author: NTUST-IB811- LIAO, YU-HUAI
"""
from cv2 import cv2
import numpy as np


# 將圖片轉灰階，再模糊化
def cv2gray(images, m=1):
    out = []
    for i in range(len(images)):
        # 圖片轉灰階
        # img = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(images[i], cv2.COLOR_BGR2HSV)[:, :, 1]
        out.append(img)

    if m == 1:
        # 回傳模糊化後圖片
        return blur_(out)

    else:
        return out


def mask(image_, mask_ratio_=0.05):
    h, w = image_.shape[:2]
    start_row = int(mask_ratio_ * h)
    start_col = int(mask_ratio_ * w)
    end_row = h - start_row
    end_col = w - start_col
    image_ = image_[start_row:end_row, start_col:end_col]

    return image_


# 模糊化
def blur_(images, m=0):
    # 設置3x3矩陣每個元素為1/9
    r = 9
    kernel = np.ones((r, r), np.float32) / (r*r)
    out = []

    if m == 1:
        for i in range(len(images)):
            # 將圖片經過3x3矩陣平均模糊化去雜訊
            dst = cv2.filter2D(images[i], -1, kernel)
            out.append(dst)

    elif m == 2:
        for i in range(len(images)):
            # 將圖片經過3x3矩陣平均模糊化去雜訊
            dst = cv2.medianBlur(images[i], r)
            out.append(dst)

    else:
        for i in range(len(images)):
            # 將圖片做高斯模糊化去雜訊
            dst = cv2.GaussianBlur(images[i], (r, r), 1)
            out.append(dst)

    return out
