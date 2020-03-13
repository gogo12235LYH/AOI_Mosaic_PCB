"""
Created on Mon Dec  10 10:50:31 2019

@author: NTUST-IB811- LIAO, YU-HUAI
"""
import sys
import pickle
import time
import numpy as np
from cv2 import cv2
from AT.Build_pts import MergeV, MergeH
from AT.Check import outputs_folder_scanner
from AT.LoadSrc import sort_name, load_images, s_sort_name_mn


class MainBD(object):
    def __init__(self, FOLDER_, NAME__, m_, n_, TH_=0.4, TH2_=0.6, KK_=500, KK2_=500):
        self.FOLDER_ = FOLDER_
        self.NAME_ = NAME__
        self.m = m_
        self.n = n_
        self.TH = TH_
        self.TH2 = TH2_
        self.KK = KK_
        self.KK2 = KK2_
        self.data_path = '../images/' + self.FOLDER_ + '/'
        self.data_o_path = '../outputs/' + self.FOLDER_ + '/'
        self.h_info = []

    def main_build(self, MODE=1, size=1):
        # 計時開始
        tic = time.time()
        print('# [ TH : %g ] [ TH2 : %g ]' % (self.TH, self.TH2))
        print('# [ KK : %d ] [ KK2 : %d ]' % (self.KK, self.KK2))

        if MODE == 1:
            outputs_folder_scanner(self.data_o_path, self.FOLDER_)
            data_img = self.CheckAndSTI_Vertical(tic, size)
            self.CheckAndSTI_Horizontal(data_img, tic)

        else:
            print('Error')

    def CheckAndSTI_Vertical(self, tic, size):
        data_img_ = []
        if self.m > 1:
            names = s_sort_name_mn(self.m, self.n, self.data_path)  # 讀取圖像名稱
            for i in range(self.n - 1, -1, -1):
                images = load_images(self.data_path, names[i], size)  # 讀取影像

                MV = MergeV(images, self.TH, self.KK)
                img, hs = MV.merge()

                data_img_.append(img)
                self.h_info.append(hs)

                # cv2.imwrite(self.data_o_path + '/z{}.png'.format(i),  img)
                # 打印目前花費時間

                print('# Time : %.3f sec\n' % (time.time() - tic))

            return data_img_

        else:
            # row(m)等於1，直接取水平拼接
            names = sort_name(self.data_path, self.m, self.n)  # 讀取圖像名稱
            images = load_images(self.data_path, names, size)  # 讀取影像
            data_img_ = images
            return data_img_

    def CheckAndSTI_Horizontal(self, data_img_, tic):
        if self.n > 1:
            MH = MergeH(data_img_, self.TH2, self.KK2)
            img_, hs = MH.merge()

            # 最終計時截止
            self.h_info.append(hs)
            toc = time.time()
            cv2.imwrite(self.data_o_path + '/' + self.NAME_ + '.png', img_)

            # save h_info
            file = open('./h_data/h_info.pickle', 'wb')
            pickle.dump(self.h_info, file)
            file.close()
            print('# Finished Time : %.3f sec' % (toc - tic))
            return img_

        else:
            toc = time.time()
            data_img_ = np.array(data_img_)
            data_img_ = np.reshape(data_img_, (data_img_.shape[1], data_img_.shape[2], 3))
            cv2.imwrite(self.data_o_path + '/' + self.NAME_ + '.png', data_img_)
            print('# Finished Time : %.3f sec' % (toc - tic))
            return data_img_


if __name__ == "__main__":
    # 資料夾名稱
    FOLDER = sys.argv[1]

    # 輸出圖片名稱
    NAME_ = sys.argv[2]

    # M x N
    m = int(sys.argv[3])  # M
    n = int(sys.argv[4])  # N

    # 閥值 0.1 ~ 1，TH = 0.35~0.6  & KK = 500 for SURF
    TH, TH2 = 0.4, 0.4

    # 特徵點閥值，值越小特徵點越多，值越大特徵點越少
    KK, KK2 = 500, 500

    main = MainBD(FOLDER,
                  NAME_, m, n,
                  TH, TH2, KK, KK2,
                  )

    main.main_build()
