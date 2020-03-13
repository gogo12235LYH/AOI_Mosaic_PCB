"""
Created on Sat Nov  9 23:25:43 2019

@author: NTUST-IB811- LIAO, YU-HUAI
"""
from cv2 import cv2
import numpy as np


class FeatureSpace(object):
    def __init__(self, img_src, img_dst, KK_, TH_):
        """
        :param img_src: Source image
        :param img_dst: Destination image
        :param KK_: SURF Threshold
        :param TH_: Scan image ratio
        """
        self.img1_ = img_src
        self.img2_ = img_dst
        self.KK = KK_
        self.TH_ = TH_
        self.ratio = 0.45
        self.d_ratio = 1 / self.ratio

    def Features(self, comp_H=0, comp_V=0):
        h1, w1 = self.img1_.shape[:2]
        h2, w2 = self.img2_.shape[:2]

        img1 = cv2.resize(self.img1_, (int(self.ratio * w1), int(self.ratio * h1)))
        img2 = cv2.resize(self.img2_, (int(self.ratio * w2), int(self.ratio * h2)))

        surf = cv2.xfeatures2d.SURF_create(self.KK, upright=True)
        # kp 及 des 為特徵點擷取後資訊
        kp1, des1 = surf.detectAndCompute(img2, None)
        kp2, des2 = surf.detectAndCompute(img1, None)

        # 建立匹配器相關功能
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        FBM_ = cv2.FlannBasedMatcher(index_params, search_params)
        # 由特徵擷取後的des進行匹配
        matches = FBM_.knnMatch(des1, des2, k=2)

        data_ = self.FindGoodAndPoints(matches, kp1, kp2, comp_H, comp_V)

        return data_

    def FindGoodAndPoints(self, matches_, kp1_, kp2_, comp_H, comp_V, MIN_MATCH_COUNT=4):
        good = []
        for m in matches_:
            if m[0].distance < self.TH_ * m[1].distance:
                good.append(m[0])
            # good.append(m[0])

        print('# Matching Points : ', len(good))

        if len(good) >= MIN_MATCH_COUNT:
            # 經由 篩選匹配結果解析 kp的 座標
            src_pts = np.float32([kp1_[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            src_pts = src_pts[:, :, :] * self.d_ratio

            # 參考圖
            dst_pts = np.float32([kp2_[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = dst_pts[:, :, :] * self.d_ratio
            dst_pts[:, :, 0] += comp_V
            dst_pts[:, :, 1] += comp_H

            dst_centriod = find_centroid(dst_pts)
            src_centriod = find_centroid(src_pts)

            src_backup = src_pts.copy()
            dst_backup = dst_pts.copy()
            lsm = build_least_squares_error(src_backup, dst_backup)
            # m = cv2.findHomography(src_pts,  dst_pts, cv2.RANSAC, 4.5)[0]
            cm_ = covariance_matrix(src_backup, dst_backup, src_centriod, dst_centriod)
            W, U, Vt = singular_value_decomposition(cm_)
            rm = build_rotation_matrix(U, Vt)
            T = find_shifting(rm, src_centriod, dst_centriod)
            print("# Shift Vector shape :", T.shape)
            print(T)

            T_NR = find_shifting_NR(src_centriod, dst_centriod)
            print(T_NR)

            # 解析好的兩組 kp 經由 RANSAC 隨機運算，得到最好的單應矩陣
            # m = cv2.findHomography(src_pts,
            #                        dst_pts,
            #                        cv2.RANSAC,
            #                        4.5  # 3x2 : 10
            #                        )[0]

            m = np.zeros((3, 3))
            # m[0, 0] = lsm[1, 0]
            # m[1, 1] = 1
            # m[2, 2] = lsm[2, 1]
            # m[0, 1] = lsm[1, 1]
            # m[1, 0] = lsm[2, 0]
            # # m[:2, :2] = rm
            # m[0, 2] = lsm[0, 0]
            # m[1, 2] = lsm[0, 1]

            m[0, 0] = 1
            m[1, 1] = 1
            m[2, 2] = 1
            m[0, 2] = T_NR[0]
            m[1, 2] = T_NR[1]

            # print(m, m.shape)

            return m

        else:
            print("# Not enough matches are found - %d / %d" % (len(good), MIN_MATCH_COUNT))
            return exit()


def find_centroid(input_point_set):
    """
    :param input_point_set: Feature point's pixel position.
    :return: Centroid from feature points.
    """
    output_point_set = np.zeros((2, 1))
    x_mean = input_point_set[:, :, 0].mean()
    y_mean = input_point_set[:, :, 1].mean()
    output_point_set[0, :] = x_mean
    output_point_set[1, :] = y_mean
    return output_point_set


def covariance_matrix(data1, data2, mean1, mean2):
    """
    :param data1: A data set
    :param data2: B data set
    :param mean1: Centroid of A data set
    :param mean2: Centroid of B data set
    :return: Covariance Matrix

    #   src_points & dst_point shape is (length, 1, 2)
    #   src_mean & dst_mean shape is (2, 1)
    """
    cm = np.zeros((2, 2))
    # print("# Shape : ", data1.shape, mean1.shape)
    for i in range(len(data1)):
        data1[:, :, 0] = data1[:, :, 0] - mean1[0]
        data1[:, :, 1] = data1[i, :, 1] - mean1[1]
        data2[:, :, 0] = data2[i, :, 0] - mean2[0]
        data2[:, :, 1] = data2[i, :, 1] - mean2[1]
        cm += np.dot(data1[i].T, data2[i])
        # cm += np.dot(data2[i].T, data1[i])
    return cm


def singular_value_decomposition(input_matrix):
    return cv2.SVDecomp(input_matrix)


def build_rotation_matrix(U, Vt_):
    """
    :param U: The matrix from Singular Value Decomposition, left orthogonal matrix
    :param Vt_: The matrix from Singular Value Decomposition, right orthogonal matrix
    :return: U dot Vt_ become rotation matrix
    """
    return np.dot(Vt_, U)


def find_shifting(rotation_matrix, centroidA, centroidB):
    print("# Find Shifting :", rotation_matrix.shape, centroidA.shape, centroidB.shape)
    tt = (np.dot(rotation_matrix, centroidA))
    return centroidB - tt


def find_shifting_NR(centroidA, centroidB):
    """
    :param centroidA:  Source image's centroid
    :param centroidB:  Destination image's centroid~
    :return: Displacement between src image and dst image
    """
    return centroidB - centroidA


def build_least_squares_error(pointX, pointY):
    """
    :param pointX: Source feature point
    :param pointY: Destination feature point
    :return: A affine matrix solve bu least squares error method
    """
    pointX = pointX.reshape((-1, 2))
    pointY = pointY.reshape((-1, 2))
    X = np.ones((pointX.shape[0], 3))
    X[:, 1:] = pointX
    # print(X.shape, pointY.shape)
    m = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), pointY)
    # print(m)
    return m
