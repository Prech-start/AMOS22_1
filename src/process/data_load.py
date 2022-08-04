import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils.file_load import *
import nibabel as nib
import pickle
from skimage.transform import resize
import SimpleITK as sitk


class data_set(Dataset):
    def __init__(self, is_train=True, is_test=False):
        self.task_json, self.path = load_json('AMOS22', 'task1_dataset.json')
        self.is_train = is_train
        if is_train:
            with open(os.path.join('..', 'checkpoints', 'tr_ts_inf', 'trainx.li_x.li'), 'rb+') as f:
                self.images_path = pickle.load(f)
            with open(os.path.join('..', 'checkpoints', 'tr_ts_inf', 'trainx.li_y.li'), 'rb+') as f:
                # pickle.load(f)
                self.labels_path = pickle.load(f)
        else:
            with open(os.path.join('..', 'checkpoints', 'tr_ts_inf', 'testx.li_x.li'), 'rb+') as f:
                self.images_path = pickle.load(f)
            with open(os.path.join('..', 'checkpoints', 'tr_ts_inf', 'testx.li_y.li'), 'rb+') as f:
                self.labels_path = pickle.load(f)
        pass

    def __len__(self):
        return len(self.images_path)
        pass

    def __getitem__(self, index):
        x = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(self.path, str(self.images_path[index], 'utf-8')))).astype(np.int16)
        y = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(self.path, str(self.labels_path[index], 'utf-8')))).astype(np.int8)
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=int)
        x = self.norm(x)
        # TODO: anti_aliasing=False 是否采用高斯滤波 label不建议使用
        # TODO: 原先使用np.round四舍五人，内存和资源消耗多，训练缓慢。
        # TODO: order=1 双线性插值 order=0邻近插值
        x = resize(x, (64, 256, 256), order=1, preserve_range=True, anti_aliasing=False)
        y = resize(y, (64, 256, 256), order=0, preserve_range=True, anti_aliasing=False)
        x = torch.from_numpy(x).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.FloatTensor)
        return x.unsqueeze_(0), y

    # TODO: norm的方法：
    #  原先： （x- min） / （x.max - x.min)
    def norm(self, x):
        x = x + 1024.0
        x = np.clip(x, a_min=0, a_max=2048.0)
        x = x / 2048.0
        return x



