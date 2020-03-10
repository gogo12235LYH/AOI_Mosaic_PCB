# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:49:36 2019

@author: ib811
"""

import os


# 檢查存放重疊區域處的資料夾
def check_fd(FOLDER_):
    path_ = '../outputs/' + FOLDER_ + '/'

    if not os.path.isdir(path_ + '01') or not os.path.isdir(path_ + '02') or not os.path.isdir(path_ + '03'):
        os.mkdir(path_ + '01')
        os.mkdir(path_ + '02')
        os.mkdir(path_ + '03')


# 檢查outputs中的資料夾
def outputs_folder_scanner(path_, FOLDER_):
    print('# Scanning folder : ')
    if not os.path.isdir(path_):
        os.mkdir(path_)
        print('# Making Folder : ', FOLDER_, ' ==> Done !')

    else:
        print('# Done !')
