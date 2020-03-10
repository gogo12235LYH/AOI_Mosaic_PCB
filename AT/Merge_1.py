"""
Created on Sat Nov  9 23:25:43 2019

@author: NTUST-IB811- LIAO, YU-HUAI
"""
from cv2 import cv2
from AT.Overlay1 import Blending
import time


class WarpDstHorizontal(object):
    def __init__(self, img1_, img2_, data_, dst1_):
        self.img1 = img1_
        self.img2 = img2_
        self.data = data_
        self.dst1 = dst1_

    def warp(self):
        h, w = self.img1.shape[:2]
        overlay = ((w - self.dst1[1]) / w) * 100
        print('# Overlay percentage : %.3f %%' % overlay)
        merge = cv2.warpPerspective(self.img2,
                                    self.data,
                                    (self.dst1[3], self.img1.shape[0]),
                                    flags=cv2.INTER_NEAREST
                                    )

        tt = time.time()
        # 顯示 參考圖 及 拼接圖 的大小資訊
        print('# Reference image shape : ', self.img1.shape)  # 參考圖
        print('# Stitching image shape : ', merge.shape)  # 拼接圖

        B = Blending(merge, self.img1, self.dst1)
        merge1 = B.BlendingHorizontalNL()

        tto = time.time()
        print('# Fix Time : %.3f sec' % (tto - tt))

        return merge1, merge


class WarpDstVertical(object):
    def __init__(self, img1_, img2_, data_, dst1_):
        self.img1 = img1_
        self.img2 = img2_
        self.data = data_
        self.dst1 = dst1_

    def warp(self):
        h, w = self.img1.shape[:2]
        overlay = ((h - self.dst1[0]) / h) * 100
        print('# Overlay percentage : %.3f %%' % overlay)
        merge = cv2.warpPerspective(self.img2,
                                    self.data,
                                    (self.dst1[3], self.dst1[2]),
                                    flags=cv2.INTER_NEAREST
                                    )

        tt = time.time()
        # 顯示 參考圖 及 拼接圖 的大小資訊
        print('# Reference image shape : ', self.img1.shape)  # 參考圖
        print('# Stitching image shape : ', merge.shape)  # 拼接圖

        B = Blending(merge, self.img1, self.dst1)
        merge1 = B.BlendingVerticalNL()

        tto = time.time()
        print('# Fix Time : %.3f sec' % (tto - tt))

        return merge1, merge
