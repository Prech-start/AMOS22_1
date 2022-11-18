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


def cal_centre_point(ori_file_path, label_file_path, r):
    import torch.nn.functional as f
    start_time = time.time()
    file_name = label_file_path.split('/')[-1].split('.')[0]
    save_path = '/home/ljc/code/AMOS22/data/pointcloud/'
    if not os.path.exists(os.path.join(save_path, '{}.npy'.format(file_name))):
        label_arr = read_nii(label_file_path)
        centre_arr = np.zeros_like(label_arr)
        for i in np.unique(label_arr).tolist():
            if i == 0:
                continue

            # class_xy = np.array(np.where(centre == 0))
            # distance_for_xyz = np.linalg.norm(class_xy.T - [10, 10], axis=1)  # N * 1
            # xy = class_xy.T[np.where(distance_for_xyz < 10)]
            # distance_list = distance_for_xyz[np.where(distance_for_xyz < 10)]
            # centre[tuple(class_xy.T[np.where(distance_for_xyz < 10)].T.tolist())] = 10 - distance_list

            class_xyz = np.array(np.where(label_arr == i))
            x, y, z = np.mean(class_xyz, 1, dtype=int)
            # 求中心点坐标
            centre_arr[x, y, z] = 1
            # 求每个label点到中心的点的距离
            distance_for_xyz = np.linalg.norm(class_xyz.T - [x, y, z], axis=1)  # N * 1
            # 筛选出distance中距离小于r的所有坐标 type:tuple
            dis = np.where(distance_for_xyz < r)
            # 筛选出到中心点距离小于r的所有坐标
            distance_list = distance_for_xyz[dis]
            # 将每个筛选出的点进行赋值
            centre_arr[tuple(class_xyz.T[dis].T.tolist())] = r - distance_list

        np.save(os.path.join(save_path, file_name), centre_arr)
        print('data {} processing is finished, it cost {}(s)'.format(file_name, time.time() - start_time))
        return centre_arr
    print('data {} processing is finished, it cost {}(s)'.format(file_name, time.time() - start_time))
    return np.load(save_path).astype(np.int8)


def cal_centre_point_2(label_arr, label_file_path, r):
    start_time = time.time()
    file_name = label_file_path.split('/')[-1].split('.')[0]
    path_dir = os.path.dirname(__file__)
    save_path = os.path.join(path_dir, '..', '..', 'data', 'pointcloud', '{}.npy'.format(file_name))
    if not os.path.exists(save_path):
        centre_arr = np.zeros_like(label_arr).astype(np.float32)
        for i in np.unique(label_arr).tolist():
            if i == 0:
                continue
            class_xyz = np.array(np.where(label_arr == i))
            x, y, z = np.mean(class_xyz, 1, dtype=int)
            # 求中心点坐标
            centre_arr[x, y, z] = 1
            # 求每个label点到中心的点的距离
            distance_for_xyz = np.linalg.norm(class_xyz.T - [x, y, z], axis=1)  # N * 1
            # 筛选出distance中距离小于r的所有坐标 type:tuple
            dis = np.where(distance_for_xyz < r)
            # 筛选出到中心点距离小于r的所有坐标
            distance_list = distance_for_xyz[dis]
            # 将每个筛选出的点进行赋值
            centre_arr[tuple(class_xyz.T[dis].T.tolist())] = r - distance_list
        np.save(save_path, centre_arr)
        print('data {} processing is finished, it cost {}(s)'.format(file_name, time.time() - start_time))
        return centre_arr
    # print('data {} processing is finished, it cost {}(s)'.format(file_name, time.time() - start_time))
    return np.load(save_path).astype(np.int8)


from src.train.loss import One_Hot


def cal_centre_point_3(label_arr, label_file_path, r):
    start_time = time.time()
    file_name = label_file_path.split('/')[-1].split('.')[0]
    path_dir = os.path.dirname(__file__)
    file_path = os.path.join(path_dir, '..', '..', 'data', 'pointcloud3')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    save_path = os.path.join(file_path, '{}.npy'.format(file_name))
    label_arr_ = torch.nn.functional.one_hot(torch.LongTensor(label_arr), 16).permute(-1, 0, 1, 2)
    if not os.path.exists(save_path):
        centre_arr = np.zeros_like(label_arr_).astype(np.float32)
        for i in np.unique(label_arr).astype(int).tolist():
            if i == 0:
                continue
            class_xyz = np.array(np.where(label_arr_[i, ...] == 1))
            class_all = np.array(np.where(label_arr_[i, ...] != -1))
            class_xyz = np.r_[np.ones((1, class_xyz.shape[-1])) * i, class_xyz].astype(int)
            class_all = np.r_[np.ones((1, class_all.shape[-1])) * i, class_all].astype(int)
            c, x, y, z = np.mean(class_xyz, 1, dtype=int)
            # 求中心点坐标
            centre_arr[i, x, y, z] = 1
            # 求每个label点到中心的点的距离
            distance_for_xyz = np.linalg.norm(class_all.T - [i, x, y, z], axis=1)  # N * 1
            # 筛选出distance中距离小于r的所有坐标 type:tuple
            dis = np.where(distance_for_xyz < r)
            # 筛选出到中心点距离小于r的所有坐标
            distance_list = distance_for_xyz[dis]
            # 将每个筛选出的点进行赋值
            centre_arr[tuple(class_all.T[dis].T.tolist())] = r - distance_list
        np.save(save_path, centre_arr)
        print('data {} processing is finished, it cost {}(s)'.format(file_name, time.time() - start_time))
        return centre_arr
    # print('data {} processing is finished, it cost {}(s)'.format(file_name, time.time() - start_time))
    return np.load(save_path).astype(np.int8)


if __name__ == '__main__':
    ori_file_path = '/home/ljc/code/AMOS22/data/AMOS22/imagesTr/amos_0001.nii.gz'
    label_file_path = '/home/ljc/code/AMOS22/data/AMOS22/labelsTr/amos_0001.nii.gz'
    # process_3d_2_pc(ori_file_path, label_file_path)
    # cal_centre_point(ori_file_path, label_file_path)
    # for ori_file_path, label_file_path in file_path:
    #     cal_centre_point(ori_file_path, label_file_path)
    x = torch.ones(64, 256, 256).long()
    cal_centre_point_3(x, label_file_path, 10)
