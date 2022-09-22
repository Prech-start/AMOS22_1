# -*-coding:utf-8 -*-

"""
# File       : trans_to_pointcloud.py
# Time       ：2022/9/19 上午9:54
# Author     ：ljc
# version    ：python 3.7
# Description：
"""
import copy
import os.path

import numpy as np
import time
import torch
import SimpleITK as sitk
from src.process.task2_sliding_window2 import file_path


def read_nii(ori_file_path):
    return np.array(sitk.GetArrayFromImage(sitk.ReadImage(ori_file_path))).astype(np.int16)


def process_3d_2_pc(ori_file_path, label_file_path):
    start_time = time.time()
    file_name = ori_file_path.split('/')[-1].split('.')[0]
    ori_arr = read_nii(ori_file_path)
    label_arr = read_nii(label_file_path)
    col = []
    for (index, value), (_, label) in zip(np.ndenumerate(ori_arr), np.ndenumerate(label_arr)):
        row = [index[0], index[1], index[2], value, label]
        col.append(row)
    col = np.array(col).astype(np.int)
    np.savetxt('/home/ljc/code/AMOS22/data/pointcloud/{}.txt'.format(file_name), X=col, fmt='%u', delimiter=' ')
    # np.save('/home/ljc/code/AMOS22/data/pointcloud/{}2'.format(file_name), col) 空间消耗过大
    print('data {} processing is finished, it cost {}(s)'.format(file_name, time.time() - start_time))


def cal_centre_point(ori_file_path, label_file_path):
    import torch.nn.functional as f
    start_time = time.time()
    file_name = label_file_path.split('/')[-1].split('.')[0]
    save_path = '/home/ljc/code/AMOS22/data/pointcloud/{}'.format(file_name)
    if not os.path.exists(save_path):
        label_arr = read_nii(label_file_path)
        centre_arr = np.zeros_like(label_arr)
        for i in np.unique(label_arr).tolist():
            if i == 0:
                continue
            x, y, z = np.mean(np.array(np.where(label_arr == i)), 1, dtype=int)
            centre_arr[x, y, z] = i
        np.save(save_path, centre_arr)
        print('data {} processing is finished, it cost {}(s)'.format(file_name, time.time() - start_time))
        return centre_arr
    print('data {} processing is finished, it cost {}(s)'.format(file_name, time.time() - start_time))
    return np.load(save_path).astype(np.int8)


def cal_centre_point_2(label_arr, label_file_path):
    start_time = time.time()
    file_name = label_file_path.split('/')[-1].split('.')[0]
    save_path = '/home/ljc/code/AMOS22/data/pointcloud/{}'.format(file_name)
    if not os.path.exists(os.path.join('/home/ljc/code/AMOS22/data/pointcloud', '{}.npy'.format(file_name))):
        centre_arr = np.zeros_like(label_arr)
        for i in np.unique(label_arr).tolist():
            if i == 0:
                continue
            x, y, z = np.mean(np.array(np.where(label_arr == i)), 1, dtype=int)
            centre_arr[x, y, z] = i
        np.save(save_path, centre_arr)
        print('data {} processing is finished, it cost {}(s)'.format(file_name, time.time() - start_time))
        return centre_arr
    print('data {} processing is finished, it cost {}(s)'.format(file_name, time.time() - start_time))
    return np.load(save_path).astype(np.int8)


if __name__ == '__main__':
    ori_file_path = '/home/ljc/code/AMOS22/data/AMOS22/imagesTr/amos_0001.nii.gz'
    label_file_path = '/home/ljc/code/AMOS22/data/AMOS22/labelsTr/amos_0001.nii.gz'
    # process_3d_2_pc(ori_file_path, label_file_path)
    # cal_centre_point(ori_file_path, label_file_path)
    for ori_file_path, label_file_path in file_path:
        cal_centre_point(ori_file_path, label_file_path)
