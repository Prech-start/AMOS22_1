import os.path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pickle
from skimage.transform import resize
import SimpleITK as sitk
import json
import sys
from src.utils.trans_to_pointcloud import cal_centre_point_2

path_dir = os.path.dirname(__file__)
task2_json = json.load(open(os.path.join(path_dir, '..', '..', 'data', 'AMOS22', 'task2_dataset.json')))
# centre_path = os.listdir('/home/ljc/code/AMOS22/data/pointcloud')
# centre_path.sort()

file_path = [[os.path.join(path_dir, '..', '..', 'data', 'AMOS22', path_['image']),
              os.path.join(path_dir, '..', '..', 'data', 'AMOS22', path_['label'])]
             for path_ in task2_json['training']]

CT_train_path = file_path[0:150]
CT_valid_path = file_path[150:160]
CT_test_path = file_path[160:200]
MRI_train_path = file_path[200:230]
MRI_valid_path = file_path[230:232]
MRI_test_path = file_path[232::]
train_path = CT_train_path + MRI_train_path
valid_path = CT_valid_path + MRI_valid_path
test_path = CT_test_path + MRI_test_path


class data_set(Dataset):
    def __init__(self, file_path, is_valid=False):
        self.paths = file_path
        self.is_valid = is_valid

    def __getitem__(self, item):
        path_ = self.paths
        x = sitk.GetArrayFromImage(sitk.ReadImage(path_[item][0])).astype(np.int16)
        y = sitk.GetArrayFromImage(sitk.ReadImage(path_[item][1])).astype(np.int8)
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=int)
        x = self.norm(x)
        if self.is_valid:
            x = resize(x, (64, 256, 256), order=1, preserve_range=True, anti_aliasing=False)
            y = resize(y, (64, 256, 256), order=0, preserve_range=True, anti_aliasing=False)
            # z = resize(z, (64, 256, 256), order=0, preserve_range=True, anti_aliasing=False)
        else:
            x = resize(x, (x.shape[0], 256, 256), order=1, preserve_range=True, anti_aliasing=False)
            y = resize(y, (y.shape[0], 256, 256), order=0, preserve_range=True, anti_aliasing=False)
            # z = resize(z, (z.shape[0], 256, 256), order=0, preserve_range=True, anti_aliasing=False)
        z = cal_centre_point_2(y.squeeze(), path_[item][1])
        x = torch.from_numpy(x).type(torch.FloatTensor).unsqueeze_(0)
        y = torch.from_numpy(y).type(torch.FloatTensor)
        z = torch.from_numpy(z).type(torch.FloatTensor)
        return x, y, z

    def __len__(self):
        return len(self.paths)

    def norm(self, x):
        if np.min(x) < 0:
            # CT 图像处理
            x = x + 1024.0
            x = np.clip(x, a_min=0, a_max=2048.0)
            x = x / 2048.0
        else:
            # MRI 图像处理
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x


def collate_fun():
    pass


def get_dataloader(is_train=True, batch_size=1):
    data = data_set(train_path if is_train else test_path)
    return DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=True if is_train else False
    )


def get_valid_data():
    data = data_set(valid_path, is_valid=True)
    return DataLoader(
        dataset=data,
        batch_size=1,
        shuffle=False
    )


if __name__ == '__main__':
    d = get_dataloader()
    for i, j, k in d:
        print(torch.unique(k))
