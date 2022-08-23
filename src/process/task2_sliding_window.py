from einops import rearrange
import os.path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pickle
from skimage.transform import resize
import SimpleITK as sitk
import json


def sliding_3D_window(image, window_size, step):
    # image.shape = b, c, d, w, h
    # step must smaller than window_size
    image = rearrange(image, ' b c d w h -> b c w h d')
    _, _, width, height, depth = image.shape
    x_step, y_step, z_step = step
    x_window_size, y_window_size, z_window_size = window_size
    is_continue = False
    for z in range(0, depth, z_step):
        if is_continue:
            continue
        for y in range(0, height - y_window_size + 1 if height > y_window_size else 1, y_step):
            for x in range(0, width - x_window_size + 1 if width > x_window_size else 1, x_step):
                # 在第三个维度上，策略采用为不抛弃任何一个像素
                # 当 window 框超出目标图像范围且上一次的window取值不是紧贴最底部时，取最底部的图像
                # 1 if z+z_window_size > depth
                # 2 if z.next() < depth -> z + step_z < depth
                # 3 step < window
                if z + z_window_size > depth and z - z_step + z_window_size != depth:
                    window = image[..., x:x_window_size + x, y:y_window_size + y, -1 - z_window_size:-1]
                    # 若取最底层的window则跳过剩余对z的迭代
                    is_continue = True
                else:
                    window = image[..., x:(x_window_size + x) if width > x_window_size else -1,
                             y:(y_window_size + y) if height > y_window_size else -1, z:z_window_size + z]
                yield window


path_dir = os.path.dirname(__file__)

task2_json = json.load(open(os.path.join(path_dir, '..', '..', 'data', 'AMOS22', 'task2_dataset.json')))

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
        # 使用slidingwindow 不使用resize
        if self.is_valid:
            x = resize(x, (64, 256, 256), order=1, preserve_range=True, anti_aliasing=False)
            y = resize(y, (64, 256, 256), order=0, preserve_range=True, anti_aliasing=False)
        x = torch.from_numpy(x).type(torch.FloatTensor).unsqueeze_(0)
        y = torch.from_numpy(y).type(torch.FloatTensor)
        return x, y

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
a = get_dataloader()
for i in a:
    print()