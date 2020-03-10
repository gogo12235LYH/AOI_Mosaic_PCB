"""
Created on Sat Nov  9 23:25:43 2019

@author: NTUST-IB811- LIAO, YU-HUAI
"""
import os


def check_fd(FOLDER_):
    path_ = '../outputs/' + FOLDER_ + '/'

    if not os.path.isdir(path + '01') or not os.path.isdir(path_ + '02') or not os.path.isdir(path_ + '03'):
        os.mkdir(path_ + '01')
        os.mkdir(path_ + '02')
        os.mkdir(path_ + '03')


def outputs_folder_scanner(path_, FOLDER_):
    print('# Scanning folder : ')
    if not os.path.isdir(path_):
        os.mkdir(path_)
        print('# Making Folder : ', FOLDER_, ' ==> Done !')

    else:
        print('# Done !')


if __name__ == '__main__':
    FOLDER = 'AA'
    path = '../outputs/' + FOLDER + '/'

    outputs_folder_scanner(path, FOLDER)
