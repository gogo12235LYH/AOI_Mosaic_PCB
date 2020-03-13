"""
Created on Sat Nov  9 23:25:43 2019

@author: NTUST-IB811- LIAO, YU-HUAI
"""
from cv2 import cv2
import time
import numpy as np
from AT.ImageProcessing import cv2gray
from AT.Feature1 import FeatureSpace
from AT.Merge_1 import WarpDstHorizontal, WarpDstVertical


class MergeH(object):
    def __init__(self, images_, TH_, KK_):
        self.images = images_
        self.TH = TH_
        self.KK = KK_
        self.t1 = 0.645
        self.t2 = 1 - self.t1
        self.merge1 = None

    def merge(self):
        h_data = []

        print('# horizontal stitching ')

        images_gray = cv2gray(self.images)

        for i in range(len(self.images) - 1):
            print('# ', i + 1)

            if i == 0:
                w1 = int(images_gray[i].shape[1] * self.t1)
                w2 = int(images_gray[i + 1].shape[1] * self.t2)

                img1 = images_gray[i][:, w1:]
                img2 = images_gray[i + 1][:, :w2]

                tic = time.time()
                fts = FeatureSpace(img1, img2, self.KK, self.TH)
                data = fts.Features(0, w1)
                print('# Scanning Features Time : %.3f sec' % (time.time() - tic))

                dst1 = TransferAndSolvePoints(self.images[i + 1], data)

                h_data.append(data)

                WDH = WarpDstHorizontal(self.images[i], self.images[i + 1], data, dst1)
                self.merge1, _ = WDH.warp()

            else:
                w1 = int(self.merge1.shape[1] * self.t1)
                w2 = int(images_gray[i + 1].shape[1] * self.t2)

                img1_ = cv2.cvtColor(self.merge1, cv2.COLOR_BGR2HSV)[:, w1:, 1]
                # img1_ = cv2.cvtColor(self.merge1, cv2.COLOR_BGR2GRAY)[:, w1:]

                img2 = images_gray[i + 1][:, :w2]

                tic = time.time()
                fts = FeatureSpace(img1_, img2, self.KK, self.TH)
                data = fts.Features(0, w1)
                print('# Scanning Features Time : %.3f sec' % (time.time() - tic))

                dst1 = TransferAndSolvePoints(self.images[i + 1], data)
                h_data.append(data)

                WDH = WarpDstHorizontal(self.merge1, self.images[i + 1], data, dst1)
                self.merge1, _ = WDH.warp()

        print('# Image stitching complete !')

        return self.merge1, h_data


class MergeV(object):
    def __init__(self, images_, TH_, KK_):
        self.images = images_
        self.TH = TH_
        self.KK = KK_
        self.t1 = 0.645
        self.t2 = 1 - self.t1
        self.merge1 = None

    def merge(self):
        h_data = []
        print('# vertical stitching ')
        images_gray = cv2gray(self.images)

        for i in range(len(self.images) - 1):
            print('#', i + 1)

            if i == 0:
                h1 = int(images_gray[i].shape[0] * self.t1)
                h2 = int(images_gray[i + 1].shape[0] * self.t2)

                img1 = images_gray[i][h1:, :]
                img2 = images_gray[i + 1][:h2, :]

                tic = time.time()
                fts = FeatureSpace(img1, img2, self.KK, self.TH)
                data = fts.Features(h1, 0)
                print('# Scanning Features Time : %.3f sec' % (time.time() - tic))

                dst1 = TransferAndSolvePoints(self.images[i + 1], data)

                h_data.append(data)
                WDV = WarpDstVertical(self.images[i], self.images[i + 1], data, dst1)
                self.merge1, _ = WDV.warp()

            else:
                h1 = int(self.merge1.shape[0] * self.t1)
                h2 = int(images_gray[i + 1].shape[0] * self.t2)

                img1_ = cv2.cvtColor(self.merge1, cv2.COLOR_BGR2HSV)[h1:, :, 1]
                # img1_ = cv2.cvtColor(self.merge1, cv2.COLOR_BGR2GRAY)[h1:, :]

                img2 = images_gray[i + 1][:h2, :]

                tic = time.time()
                fts = FeatureSpace(img1_, img2, self.KK, self.TH)
                data = fts.Features(h1, 0)
                print('# Scanning Features Time : %.3f sec' % (time.time()-tic))

                dst1 = TransferAndSolvePoints(self.images[i + 1], data)

                h_data.append(data)
                WDV = WarpDstVertical(self.merge1, self.images[i + 1], data, dst1)
                self.merge1, _ = WDV.warp()

        print('# Image stitching complete !')

        return self.merge1, h_data


def TransferAndSolvePoints(image_, data_):
    h, w = image_.shape[:2]
    pts_ = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst_ = cv2.perspectiveTransform(pts_, data_)

    start_row = int(dst_[:, :, 1].min())
    start_col = int(dst_[:, :, 0].min())
    end_row = int(dst_[:, :, 1].max())
    end_col = int(dst_[:, :, 0].max())
    start_sec_col = int(dst_[:2, :, 0].max())

    return start_row, start_col, end_row, end_col, start_sec_col
