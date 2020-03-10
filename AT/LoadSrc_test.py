"""
Created on Sat Nov  9 23:25:43 2019

@author: NTUST-IB811- LIAO, YU-HUAI
"""
from cv2 import cv2
import os
import numpy as np
import time


# 路徑取檔
def load_images(path_, names, size=1):
    images = []
    print('# Loading images : ', names)

    for name_ in names:
        img = cv2.imread(path_ + name_)
        h, w = img.shape[:2]
        # 可控制圖片大小
        img = cv2.resize(img, (int(w * size), int(h * size)))
        images.append(img)

    return images


# 正常讀取資料名稱
def normal_name(path_, size):
    # 根據路徑讀取資料夾內所有檔案名稱
    image_name = os.listdir(path_)
    return load_images(path_, image_name, size)


# 排列讀取資料名稱
def sort_name(path_, iff):
    # 根據路徑讀取資料夾內所有檔案名稱
    image_name = os.listdir(path_)
    image_name.sort(key=lambda x: str(x[:-4]))
    images = []

    if iff == 1:
        images = image_name
        print('# Find images : ', image_name)
        print('# G1 : ', images)

    else:
        # 儲存一反向的名稱
        image_reverse = image_name.copy()
        image_reverse.reverse()

        print('# Find images : ', image_name)

        # 計算垂直向的圖片數量
        length = int(len(image_name) / iff)

        num_ = iff - 1

        # 目前機台可能上限為 水平拍照最多2張，垂直拍照至少2張
        for i in range(iff):
            if i == 0:
                images.append(image_name[num_:num_ + length])
            else:
                images.append(list(np.concatenate((image_name[:i], image_reverse[:length - 1]))))

        for p in range(len(images)):
            print('# G{} : '.format(p + 1), images[p])

    return images


def sort_name_mn(n, m, path_):  # MxN
    image_name = os.listdir(path_)
    image_name.sort(key=lambda x: str(x[:-4]))
    z = [[None for i_ in range(n)] for j_ in range(m)]

    print('# Find images : ', image_name)

    cnt = 0
    s = m - n
    d = s + 1

    if m == n or n < 3:
        w = 0
    elif n >= 3:
        w = s

    for i in range(int(n / 2)):
        #
        try:
            for j in range(int(n - i)):
                if z[i][j] is None:
                    z[i][j] = image_name[cnt]
                    cnt += 1
        except:
            print(z)

        #
        try:
            for j in range(i + 1, m - i):
                if z[j][m - i - d] is None:
                    z[j][m - i - d] = image_name[cnt]
                    cnt += 1
        except:
            print(z)
        #
        try:
            for j in range(m - i - d, i, -1):
                if z[m - d - i + w][j] is None:
                    z[m - d - i + w][j] = image_name[cnt]
                    cnt += 1
        except:
            print(z)
        #
        try:
            for j in range(m - i - d + s, i, -1):
                if z[j][i] is None:
                    z[j][i] = image_name[cnt]
                    cnt += 1
        except:
            print(z)

    if n % 2 == 1 and m == n:
        if z[int(n / 2)][int(n / 2)] is None:
            z[int(n / 2)][int(n / 2)] = image_name[cnt]

    elif n % 2 == 1:
        for _ in range(s + 1):
            if z[int(n / 2) + _][int(n / 2)] is None:
                z[int(n / 2) + _][int(n / 2)] = image_name[cnt]
                cnt += 1
        # if z[int(n / 2) + 1][int(n / 2)] is None:
        #     z[int(n / 2) + 1][int(n / 2)] = image_name[cnt]

    z = np.rot90(z, 1)

    for p in range(len(z)):
        print('# G{} : '.format(p + 1), z[p])

    return z


def s_sort_name_mn(m, n, path_):  # MxN
    image_name = os.listdir(path_)
    image_name.sort(key=lambda x: str(x[:-4]))
    z = [[None for i_ in range(n)] for j_ in range(m)]

    cnt = 0
    print('# Find images : ', image_name)

    for row in range(m):
        if row % 2 == 1:
            for col in range(n-1, -1, -1):
                if z[row][col] is None:
                    z[row][col] = image_name[cnt]
                    cnt += 1
        else:
            for col in range(n):
                if z[row][col] is None:
                    z[row][col] = image_name[cnt]
                    cnt += 1
    print(z)
    z = np.rot90(z, 1)
    for p in range(len(z)):
        print('# G{} : '.format(p + 1), z[p])

    return z


def sort_name_mn_cnt(n, m, ):  # MxN
    z = np.zeros((m, n), dtype=np.int)

    cnt = 1
    s = m - n
    d = s + 1
    w = 0

    if m == n or n < 3:
        w = 0
    elif n >= 3:
        w = s

    for i in range(int(n / 2)):
        #
        try:
            for j in range(int(n - i)):
                if z[i, j] == 0:
                    z[i, j] = cnt
                    cnt += 1
        except:
            print(z)

        #
        try:
            for j in range(i + 1, m - i):
                if z[j, m - i - d] == 0:
                    z[j, m - i - d] = cnt
                    cnt += 1
        except:
            print(z)
        #
        try:
            for j in range(m - i - d, i, -1):
                if z[m - d - i + w, j] == 0:
                    z[m - d - i + w, j] = cnt
                    cnt += 1
        except:
            print(z)
        #
        try:
            for j in range(m - i - d + s, i, -1):
                if z[j, i] == 0:
                    z[j, i] = cnt
                    cnt += 1
        except:
            print(z)

    if n % 2 == 1 and m == n:
        if z[int(n / 2), int(n / 2)] == 0:
            z[int(n / 2), int(n / 2)] = cnt

    elif n % 2 == 1:
        for _ in range(s + 1):
            if z[int(n / 2) + _, int(n / 2)] == 0:
                z[int(n / 2) + _, int(n / 2)] = cnt
                cnt += 1

    z = np.rot90(z, 1)

    for p in range(len(z)):
        print('# G{} : '.format(p + 1), z[p])

    return z


def s_sort_name_mn_cnt(m, n, ):  # MxN
    z = np.zeros((m, n), dtype=np.int)
    cnt = 1

    for row in range(m):
        if row % 2 == 1:
            for col in range(n-1, -1, -1):
                if z[row, col] == 0:
                    z[row, col] = cnt
                    cnt += 1
        else:
            for col in range(n):
                if z[row, col] == 0:
                    z[row, col] = cnt
                    cnt += 1
    print(z)

    for p in range(len(z)):
        print('# G{} : '.format(p + 1), z[p])

    return z


if __name__ == '__main__':
    path = 'D:\\aoi_test\\images\\test3\\'
    tic = time.time()
    # name = sort_name_mn(5, 6, path)
    # name = sort_name_mn_cnt(1000, 1000)
    # name = s_sort_name_mn_cnt(2, 3)
    name = s_sort_name_mn(3, 1, path)
    toc = time.time()

    print("# Time : %.5f sec " % (toc - tic))
