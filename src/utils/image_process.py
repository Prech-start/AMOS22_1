import copy

import torch

import PIL.Image as Image
import numpy as np
from src.model.model import *
import os
import nibabel as nib
from einops import rearrange
from src.process.data_load import *
from torch.utils.data import DataLoader
import skimage


def norm(x):
    if np.min(x) < 0:
        # CT 图像处理
        x = x + 1024.0
        x = np.clip(x, a_min=0, a_max=2048.0)
        x = x / 2048.0
    else:
        # MRI 图像处理
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def trans_3d_2_mp4(image_array, result_array):
    image_array = image_array.data.squeeze().cpu().numpy()
    result_array = result_array.data.squeeze().cpu().numpy()
    image_array = norm(image_array) * 255
    depth, height, width = image_array.shape
    palettedata = [0, 0, 0, 102, 0, 255, 0, 255, 176, 51, 255, 204, 184, 138, 0, 255, 102, 51, 102, 51, 255, 51, 255,
                   102, 153, 51, 102, 102, 51, 153, 255, 20, 20, 20, 255, 255, 194, 10, 255, 51, 51, 153, 255, 255, 61,
                   255, 0, 128]
    tar_images = []
    for d in range(depth):
        image_fps, result_fps = image_array[d], result_array[d]
        image_fps = Image.fromarray(image_fps.astype('uint8'), 'L').convert('RGB')
        result_fps = Image.fromarray(result_fps.astype('uint8'), 'P')
        result_fps.putpalette(palettedata)
        result_fps = result_fps.convert('RGB')
        tar_images.append(Image.blend(image_fps, result_fps, 0.3))

    tar_images[0].save('array.gif', save_all=True, append_images=tar_images[1:], duration=200, loop=1)
    import moviepy.editor as mp
    temp = mp.VideoFileClip('array.gif')
    temp.write_videofile('result.mp4')
    os.remove('array.gif')


origin_image_path = '../utils/amos_0008.nii.gz'
label_image_path = '../utils/amos_0008I.nii.gz'
image = np.array(sitk.GetArrayFromImage(sitk.ReadImage(origin_image_path)).astype(np.int16))
label = np.array(sitk.GetArrayFromImage(sitk.ReadImage(label_image_path)).astype(np.int8))
image = torch.from_numpy(image).type(torch.FloatTensor)
label = torch.from_numpy(label).type(torch.FloatTensor)
trans_3d_2_mp4(image, label)


def overlap(images, outputs, slice=1 / 3):
    images_ori = images.data.squeeze().cpu().numpy()[images.shape[-3] * slice, ...]
    images_ori = np.expand_dims(images_ori, axis=-1)
    images_ori = array_to_img(images_ori)
    images_ori = images_ori.convert("RGB")
    image_mask = outputs.data.squeeze().cpu().numpy()[outputs.shape[-3] * slice, ...]
    image_mask = Image.fromarray(image_mask.astype('uint8'), 'P')
    palettedata = [0, 0, 0, 102, 0, 255, 0, 255, 176, 51, 255, 204, 184, 138, 0, 255, 102, 51, 102, 51, 255, 51, 255,
                   102, 153, 51, 102, 102, 51, 153, 255, 20, 20, 20, 255, 255, 194, 10, 255, 51, 51, 153, 255, 255, 61,
                   255, 0, 128]
    image_mask.putpalette(palettedata)
    image_mask = image_mask.convert('RGB')
    img = Image.blend(images_ori, image_mask, 0.7)  # blend_img = img1 * (1 – 0.3) + img2* alpha
    img.save('..' + '/result_overlap/pt_{}_compare_{}.png'.format(1, 2))


def array_to_img(x, scale=True):
    # target PIL image has format (height, width, channel) (512,512,1)
    x = np.asarray(x, dtype=float)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image).'
                         'Got array with shape', x.shape)
    if scale:
        x = x + max(-np.min(x), 0)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number:', x.shape[2])


def bind(a, b):
    '''
        a shape -> b 1 d w h
        b shape -> b 1 d w h
        ori_image -> ground truth
        mask_image -> pred
    '''
    a = a.data.squeeze().cpu().numpy()
    b = b.data.squeeze().cpu().numpy()
    ori_image = np.expand_dims(a[int(a.shape[0] / 2), :, :], -1)
    mask_image = np.expand_dims(b[int(b.shape[0] / 2), :, :], -1)
    # ori_image = rearrange(ori_image, 'w h -> w h c')
    # mask_image = rearrange(mask_image, 'w h -> w h c')
    ori_image = array_to_img(ori_image)
    mask_image = array_to_img(mask_image)
    palettedata = [0, 0, 0, 102, 0, 255, 0, 255, 176, 51, 255, 204, 184, 138, 0, 255, 102, 51, 102, 51, 255, 51, 255,
                   102, 153, 51, 102, 102, 51, 153, 255, 20, 20, 20, 255, 255, 194, 10, 255, 51, 51, 153, 255, 255, 61,
                   255, 0, 128]
    ori_image.putpalette(palettedata)
    ori_image = ori_image.convert('RGB')
    mask_image.putpalette(palettedata)
    mask_image = mask_image.convert('RGB')
    img = Image.blend(ori_image, mask_image, 0.7)  # blend_img = img1 * (1 – 0.3) + img2* alpha
    img.save('..' + '/result_overlap/pt_{}_compare_{}.png'.format(1, 2))


def show_two(a, b, file_name, slices=1.0 / 2):
    a = a.data.squeeze().cpu().numpy()
    b = b.data.squeeze().cpu().numpy()
    ori_image = np.expand_dims(a[int(a.shape[0] * slices), :, :], -1)
    mask_image = np.expand_dims(b[int(b.shape[0] * slices), :, :], -1)
    img_show = np.concatenate((ori_image, mask_image), axis=0)
    image_show = Image.fromarray(img_show[:, :, 0].astype('uint8'), 'P')
    palettedata = [0, 0, 0, 102, 0, 255, 0, 255, 176, 51, 255, 204, 184, 138, 0, 255, 102, 51, 102, 51, 255, 51, 255,
                   102, 153, 51, 102, 102, 51, 153, 255, 20, 20, 20, 255, 255, 194, 10, 255, 51, 51, 153, 255, 255, 61,
                   255, 0, 128]
    image_show.putpalette(palettedata)
    image_show = image_show.convert('RGB')
    if file_name != '' or None:
        image_show.save('..' + '/result_overlap/{}.png'.format(file_name))
    else:
        image_show.save('..' + '/result_overlap/pt_{}_compare_{}.png'.format(1, 2))


import SimpleITK as sitk


# 按照文件列表保存
def save_image_information():
    path = os.path.join('..', '..', 'data', 'AMOS22')
    with open(os.path.join('..', 'checkpoints', 'tr_ts_inf', 'testx.li_x.li'), 'rb+') as f:
        image_path = pickle.load(f)
    model = UnetModel(1, 16, 6)
    model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'Unet-210.pth')))
    for file_path in image_path:
        ori_image = sitk.ReadImage(os.path.join(path, bytes.decode(file_path)))
        model.cpu()
        x = copy.deepcopy(ori_image)
        x = np.array(sitk.GetArrayFromImage(x))
        shape_ = x.shape
        x = resize(x, (64, 256, 256), order=1, preserve_range=True, anti_aliasing=False)
        x = torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
        x = data_set().norm(x)
        pred = model(x)
        pred = torch.argmax(pred, dim=1)
        pred = pred.data.cpu().squeeze().numpy()
        pred_array = resize(pred, shape_, order=0, preserve_range=True, anti_aliasing=False)
        result_image = sitk.GetImageFromArray(pred_array)
        result_image.CopyInformation(ori_image)
        sitk.WriteImage(result_image,
                        os.path.join('..', 'output', 'predict', '{}'.format(str(file_path)[:-1].rsplit('/', 1)[1])))
        print()


COL = 3
ROW = 1
HEIGHT = 256
WIDTH = 256


def concat_image(ORI, GT, PRED, save_path='', no=0, slices=1.0 / 3):
    '''
        ORI shape = b,c,d,w,h
        G T shape = b,c,d,w,h
        PRED shape = b,c,d,w,h
    '''
    HEIGHT = ORI.shape[-1]
    WIDTH = ORI.shape[-2]
    ORI = trans_image(ORI, slices=slices, mode='L')
    GT = trans_image(GT, slices=slices, mode='P')
    PRED = trans_image(PRED, slices=slices, mode='P')
    palettedata = [0, 0, 0, 102, 0, 255, 0, 255, 176, 51, 255, 204, 184, 138, 0, 255, 102, 51, 102, 51, 255, 51, 255,
                   102, 153, 51, 102, 102, 51, 153, 255, 20, 20, 20, 255, 255, 194, 10, 255, 51, 51, 153, 255, 255, 61,
                   255, 0, 128]
    GT.putpalette(palettedata)
    PRED.putpalette(palettedata)
    GT = GT.convert('RGB')
    PRED = PRED.convert('RGB')
    target = Image.new('RGB', (WIDTH * COL, HEIGHT * ROW))
    image_files = [ORI, GT, PRED]
    for row in range(ROW):
        for col in range(COL):
            target.paste(image_files[COL * row + col], (0 + WIDTH * col, 0 + HEIGHT * row))
    if save_path != '' or None:
        target.save('..' + '/result_overlap/{}.png'.format(save_path))
    else:
        target.save('..' + '/result_overlap/pt{}_ori_compare_gt_and_pred.png'.format(no))


def trans_image(x, slices, mode="P"):
    # 删除为一的维度 batchsize，channel
    x = x.data.squeeze().cpu().numpy()
    # 将d维度转移到最后一维
    x = np.expand_dims(x[int(x.shape[0] * slices), :, :], -1)
    # 将ORI,GT,PRED转化为PIL的image
    x = Image.fromarray(x[:, :, 0].astype('uint8'), mode=mode)
    return x


# a = np.ones(shape=(1, 1, 56, 224, 224))
# b = np.zeros(shape=(1, 1, 56, 224, 224))
# a = torch.Tensor(a)
# b = torch.Tensor(b)
# show_two(a, b, 'test')
# save_image_information()
import SimpleITK as sitk

# model = UnetModel(1, 16, 6)
# model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'Unet-210.pth')))
# ori_image = sitk.ReadImage('amos_0573.nii.gz')
# model.cpu()
# x = copy.deepcopy(ori_image)
# x = np.array(sitk.GetArrayFromImage(x))
# shape_ = x.shape
# x = resize(x, (64, 256, 256), order=1, preserve_range=True, anti_aliasing=False)
# x = torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
# x = norm(x)
# pred = model(x)
# pred = torch.argmax(pred, dim=1)
# pred = pred.data.cpu().squeeze().numpy()
# pred_array = resize(pred, shape_, order=0, preserve_range=True, anti_aliasing=False)
# result_image = sitk.GetImageFromArray(pred_array)
# result_image.CopyInformation(ori_image)
# sitk.WriteImage(result_image, '0573.nii.gz')
from src.process.task2_sliding_window2 import train_path

# for p_ in train_path:
#     x_path, y_path = p_
#     x_array = np.array(sitk.GetArrayFromImage(sitk.ReadImage(x_path)).astype(np.int16))
#     y_array = np.array(sitk.GetArrayFromImage(sitk.ReadImage(y_path)).astype(np.int16))
#     x_tensor = torch.from_numpy(x_array).type(torch.FloatTensor)
#     y_tensor = torch.from_numpy(y_array).type(torch.FloatTensor)
#     # concat_image(x_tensor, y_tensor, y_tensor)
#     overlap(x_tensor, y_tensor)
#     pass
