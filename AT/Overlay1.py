"""
Created on Sat Nov  9 23:25:43 2019

@author: NTUST-IB811- LIAO, YU-HUAI
"""
import numpy as np
import numba as nb


class Blending(object):

    def __init__(self, merge_, img1_, dst1_):
        self.merge = merge_
        self.merge1 = self.merge.copy()
        self.img1 = img1_
        self.dst1 = dst1_
        self.h = self.img1.shape[0]
        self.w = self.img1.shape[1]

    def BlendingHorizontalNL(self):
        return BlendingHorizontal_Normal(self.merge, self.merge1, self.img1, self.dst1, self.h, self.w)

    def BlendingVerticalNL(self):
        return BlendingVertical_Normal(self.merge, self.merge1, self.img1, self.dst1, self.h, self.w)

    def BlendingHorizontalTri(self):
        return BlendingHorizontal_Tri(self.merge, self.merge1, self.img1, self.dst1, self.h, self.w)

    def BlendingVerticalTri(self):
        return BlendingVertical_Tri(self.merge, self.merge1, self.img1, self.dst1, self.h, self.w)


@nb.njit(nogil=True)
def BlendingHorizontal_Tri(merge_, merge1_, img1_, dst1_, h_, w_):
    # 將參考圖填補回拼接圖
    for ch in range(3):
        for ro in range(img1_.shape[0]):
            for co in range(img1_.shape[1]):
                merge1_[ro, co, ch] = img1_[ro, co, ch]

    # 處理水平方向重疊處
    for ch in range(3):
        for c in range(dst1_[4], w_):
            for r in range(h_):
                q = (np.pi * (w_ - c)) / (2 * (w_ - dst1_[4]))

                # w_1 及 w_2 為權重
                w_1 = np.square(np.cos(q))
                w_2 = np.square(np.sin(q))

                # 三角權重改善演算法實現
                merge1_[r, c, ch] = w_2 * img1_[r, c, ch] + w_1 * merge_[r, c, ch]
    return merge1_


@nb.njit(nogil=True)
def BlendingHorizontal_Normal(merge_, merge1_, img1_, dst1_, h_, w_):
    # 將參考圖填補回拼接圖
    for ch in range(3):
        for ro in range(img1_.shape[0]):
            for co in range(img1_.shape[1]):
                merge1_[ro, co, ch] = img1_[ro, co, ch]

    # 處理水平方向重疊處
    # for ch in range(3):
    #     for c in range(dst1_[4], w_):
    #         for r in range(h_):
    #             # w_1 及 w_2 為權重
    #             d_w = (w_ - dst1_[4])**3
    #             d_1 = (w_ - c)**3
    #             w_2 = d_1 / d_w
    #             w_1 = 1 - w_2
    #
    #             # 漸入漸出演算法實現
    #             merge1_[r, c, ch] = w_2 * img1_[r, c, ch] + w_1 * merge_[r, c, ch]
    return merge1_


@nb.njit(nogil=True)
def BlendingVertical_Tri(merge_, merge1_, img1_, dst1_, h_, w_):
    # 限定範圍(*)
    if w_ > dst1_[3]:
        w_ = dst1_[3]

    if h_ > dst1_[2]:
        h_ = dst1_[2]

    # 將參考圖填補回拼接圖
    for ch in range(3):
        for ro in range(img1_.shape[0]):
            for co in range(img1_.shape[1]):
                merge1_[ro, co, ch] = img1_[ro, co, ch]

    # 處理縱向重疊處
    for ch in range(3):
        for r in range(dst1_[0], h_):
            for c in range(w_):
                q = (np.pi * (h_ - r)) / (2 * (h_ - dst1_[0]))

                # w_1 及 w_2 為權重
                w_1 = np.square(np.cos(q))
                w_2 = np.square(np.sin(q))

                # 三角權重改善演算法實現
                merge1_[r, c, ch] = w_2 * img1_[r, c, ch] + w_1 * merge_[r, c, ch]

    return merge1_


@nb.njit(nogil=True)
def BlendingVertical_Normal(merge_, merge1_, img1_, dst1_, h_, w_):
    # 限定範圍(*)
    if w_ > dst1_[3]:
        w_ = dst1_[3]

    if h_ > dst1_[2]:
        h_ = dst1_[2]

    # 將參考圖填補回拼接圖
    for ch in range(3):
        for ro in range(img1_.shape[0]):
            for co in range(img1_.shape[1]):
                merge1_[ro, co, ch] = img1_[ro, co, ch]

    # 處理縱向重疊處
    # for ch in range(3):
    #     for r in range(dst1_[0], h_):
    #         for c in range(w_):
    #             # w_1 及 w_2 為權重
    #             d_w = (h_ - dst1_[0])**3
    #             d_1 = (h_ - r)**3
    #             w_1 = d_1 / d_w
    #             w_2 = 1 - w_1
    #
    #             # 漸入漸出演算法實現
    #             merge1_[r, c, ch] = w_1 * img1_[r, c, ch] + w_2 * merge_[r, c, ch]
    return merge1_
