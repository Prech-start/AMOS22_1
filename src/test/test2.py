import matplotlib
from tqdm import tqdm
matplotlib.use('TkAgg')
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
example_filename = '../output/out.nii.gz'
a = '/home/ljc/code/AMOS22/data/AMOS22/imagesTr/amos_0600.nii.gz'
# img = nib.load(a)
# # print(img)
# # print(img.header['db_name'])  # 输出头信息
# width, height, queue = img.dataobj.shape
# img = img.get_fdata()
# OrthoSlicer3D(img).show()
import cv2
import nibabel as nib
import numpy as np
from PIL import Image
from src.process.data_load import *
import pickle
import torch
import numpy as np



def save_filename(dataset, path):
    path = path + 'x.li'
    x_file_list=[]
    y_file_list = []
    for _, x_name, y_name in tqdm(dataset):
        x_file_list.append(x_name.encode('utf-8'))
        y_file_list.append(y_name.encode('utf-8'))
    pickle.dump(np.array(x_file_list), open(path + '_x.li', 'wb+'))
    pickle.dump(np.array(y_file_list), open(path + '_y.li', 'wb+'))


if __name__ == '__main__':
    # show_nii_gif(nii_file=example_filename)

    whole_set = data_set()
    lens = len(whole_set)
    train_len = lens * 0.8
    train_set, test_set = torch.utils.data.random_split(whole_set, [int(train_len), lens - int(train_len)],
                                                        torch.Generator().manual_seed(0))
    print('processing... train')
    save_filename(train_set, os.path.join('..', 'checkpoints', 'tr_ts_inf', 'train'))
    print('processing... test')
    save_filename(test_set, os.path.join('..', 'checkpoints', 'tr_ts_inf', 'test'))




