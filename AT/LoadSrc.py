from cv2 import cv2
import os
import numpy as np


# 路徑取檔
def load_images(path, names, size=1):
    images = []
    print('# Loading images : ', names)
    for name in names:
        img = cv2.imread(path + name)
        h, w = img.shape[:2]
        # 可控制圖片大小
        img = cv2.resize(img, (int(w * size), int(h * size)))
        images.append(img)

    return images


# 正常讀取資料名稱
def normal_name(path, size):
    # 根據路徑讀取資料夾內所有檔案名稱
    image_name = os.listdir(path)

    return load_images(path, image_name, size)


# 排列讀取資料名稱
def sort_name(path, iff, iff1):
    # 根據路徑讀取資料夾內所有檔案名稱
    image_name = os.listdir(path)
    images = []
    if iff == 1:
        for i in range(iff1):
            images.append(image_name[i])
    else:
        images = image_name

    # 打印分組拼接的影像名稱
    print('# Find images : ', image_name)
    print('# G1 : ', images)
    return images


# S 行路徑讀取影像資料
def s_sort_name_mn(m, n, path):  # MxN
    image_name = os.listdir(path)
    # 影像名稱從小到大排列(防止出錯)
    image_name.sort(key=lambda x: str(x[:-4]))

    # 建立 根據 M 及 N ，產生一 M x N大小的矩陣
    z = [[None for i_ in range(n)] for j_ in range(m)]

    cnt = 0  # 計數用(從0開始)
    print('# Find images : ', image_name)

    # 最外層迴圈控制 row ， 內層迴圈控制 column
    for row in range(m):
        # 偶數 row 需要反向填寫
        if row % 2 == 1:
            for col in range(n - 1, -1, -1):
                if z[row][col] is None:
                    z[row][col] = image_name[cnt]
                    cnt += 1

        # 奇數 row 正向填寫
        else:
            for col in range(n):
                if z[row][col] is None:
                    z[row][col] = image_name[cnt]
                    cnt += 1

    # 矩陣旋轉90度(逆時針)，用意在於分組拼接(垂直拼接以及水平拼接)
    z = np.rot90(z, 1)

    # 打印分組拼接的影像名稱
    for p in range(len(z)):
        print('# G{} : '.format(p + 1), z[p])

    return z
