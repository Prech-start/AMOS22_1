import numpy as np
from PIL.Image import Image
from src.utils.image_process import *
import torch

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
    # # 删除为一的维度 batchsize，channel
    # ORI = ORI.data.squeeze().cpu().numpy()
    # GT = GT.data.squeeze().cpu().numpy()
    # PRED = PRED.data.squeeze().cpu().numpy()
    # # 将d维度转移到最后一维 
    # ORI = np.expand_dims(ORI[int(ORI.shape[0] * slices), :, :], -1)
    # GT = np.expand_dims(GT[int(GT.shape[0] * slices), :, :], -1)
    # PRED = np.expand_dims(PRED[int(PRED.shape[0] * slices), :, :], -1)
    # # 将ORI,GT,PRED转化为PIL的image
    # Image.fromarray(img_show[:, :, 0].astype('uint8'), 'P')
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


# a = torch.Tensor(np.random.randint(0, 15,(1, 1, 64, WIDTH, HEIGHT)))
# concat_image(a, a, a)
from src.utils.train_utils import *

pic_loss_line()
print('0')
