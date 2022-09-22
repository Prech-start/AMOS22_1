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

# def show_nii_gif(nii_file, save_path="array.gif", show_frame=True):
#     """
#     @Brife:
#         将 shape=(arr_len, h, w) 的 nii 文件转化为 gif
#     @Param:
#         nii_file   : nii 文件的路径
#         show_frame : 是否展示这是第几帧, 默认 True
#         save_path  : 默认为当前路径的 array.gif 文件
#     """
#
#     # 调包加载 nii 文件
#     img = nib.load(nii_file)
#     # 转化成 numpy.ndarray 的格式
#     img_arr = img.get_fdata()
#     img_arr = np.squeeze(img_arr)
#
#     # 找到最大最小值, 方便之后归一化
#     img_max, img_min = img_arr.max(), img_arr.min()
#
#     # 归一化
#     img_arr = (img_arr - img_min) / (img_max - img_min) * 255
#     img_arr = img_arr.astype(np.uint8)
#
#     # 将单通道的灰度图转化为RGB三通道的灰色图, (不转化这一步, 没法写字)
#     img_RGB_list = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_arr]
#
#     assert 3 == len(img_arr.shape)  # 如果是别的shape的, 那处理不了
#
#     arr_len, h, w = img_arr.shape
#
#     if show_frame:
#         # 在每一帧上打上这是第几帧
#         for i in range(arr_len):
#             img_RGB_list[i] = cv2.putText(img_RGB_list[i],
#                                           '{:>03d}/{:>03d}'.format(i + 1, arr_len),
#                                           (50, 50),
#                                           cv2.FONT_HERSHEY_COMPLEX,
#                                           1,
#                                           (255, 255, 255),
#                                           2)
#
#     # 将所有的 ndarray 转化为 PIL.Image 格式
#     imgs = [Image.fromarray(img) for img in img_RGB_list]
#
#     # 保存
#     # duration is the number of milliseconds between frames; this is 40 frames per second
#     imgs[0].save(save_path, save_all=True, append_images=imgs[1:], duration=50, loop=0)


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




