import SimpleITK as sitk
import numpy as np
from image_process import concat_image2


def cut_(ori_path, gt_path):
    ori_image = np.array(sitk.GetArrayFromImage(sitk.ReadImage(ori_path)))
    gt_image = np.array(sitk.GetArrayFromImage(sitk.ReadImage(gt_path)))
    ori_image = norm(ori_image)
    concat_image2(ori_image * 255, gt_image, gt_path, 48)


def norm(x):
    if np.min(x) < 0:
        # CT 图像处理
        x = np.clip(x, a_min=-175, a_max=250)
        x = (x + 175) / 425
    else:
        # MRI 图像处理
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


if __name__ == '__main__':
    ori_path, gt_path = '/home/ljc/code/AMOS22/data/AMOS22/imagesTr/amos_0001.nii.gz', '/home/ljc/code/AMOS22/data/AMOS22/labelsTr/amos_0001.nii.gz'
    cut_(ori_path, gt_path)
