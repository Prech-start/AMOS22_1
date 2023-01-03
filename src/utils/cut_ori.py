import os.path
import pickle
import time

import SimpleITK as sitk
import numpy as np


def cut_(ori_path, gt_path, save_path, spacing=None):
    '''
    para:
        ori_path, gt_path : root path of ori_image and gt_image
        save_path : root path of cropped image's folder
    '''
    start_time = time.time()
    ori_image = np.array(sitk.GetArrayFromImage(sitk.ReadImage(ori_path)))
    gt_image = np.array(sitk.GetArrayFromImage(sitk.ReadImage(gt_path)))
    if np.min(ori_image) < 0:
        pre_cut_ori_image = norm(ori_image).squeeze()
    else:
        pre_cut_ori_image = ori_image.squeeze()
    mask = get_mask(pre_cut_ori_image)
    background_pixel_value = np.min(ori_image)
    ori_image *= mask
    # gt_image *= mask
    ori_image[mask == 0] = background_pixel_value
    # gt_image[mask == 0] = background_pixel_value
    x_list, y_list, z_list = np.where(mask > 0)
    x_max, y_max, z_max = np.max(x_list), np.max(y_list), np.max(z_list)
    x_min, y_min, z_min = np.min(x_list), np.min(y_list), np.min(z_list)
    print('from shape:{}'.format(ori_image.shape))
    cut_ori_image = ori_image[x_min:x_max, y_min:y_max, z_min:z_max]
    print('to {} '.format(cut_ori_image.shape))
    cut_gt_image = gt_image[x_min:x_max, y_min:y_max, z_min:z_max]
    spacing = spacing if not spacing == None else sitk.ReadImage(ori_path).GetSpacing()
    information = [spacing, x_max, y_max, z_max, x_min, y_min, z_min]
    import save_nii
    ori_file_name = ori_path.split('/')[-1]
    gt_file_name = gt_path.split('/')[-1]
    if not os.path.exists(os.path.join(save_path, 'inf')) or not os.path.exists(
            os.path.join(save_path, 'tr_ori')) or not os.path.exists(os.path.join(save_path, 'tr_gt')):
        os.makedirs(os.path.join(save_path, 'inf'))
        os.makedirs(os.path.join(save_path, 'tr_ori'))
        os.makedirs(os.path.join(save_path, 'tr_gt'))
    save_nii.np2nii(cut_ori_image, spacing=information[0], outDir=os.path.join(save_path, 'tr_ori', ori_file_name))
    save_nii.np2nii(cut_gt_image, spacing=information[0], outDir=os.path.join(save_path, 'tr_gt', gt_file_name))
    pickle.dump(information,
                open(os.path.join(save_path, 'inf', '{}.inf'.format(gt_file_name.split('.')[0])), 'wb+'))
    print('processed: from xx shape to xx shape, cost {} s.'.format(time.time() - start_time))


def norm(x):
    if np.min(x) < 0:
        # CT 图像处理
        x = np.clip(x, a_min=-175, a_max=250)
        x = (x + 175) / 425
    else:
        # MRI 图像处理
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def get_mask(img: np.ndarray):
    img_np = np.array(img)
    label_np = np.zeros_like(img_np).astype(np.uint8)
    label_np[img_np > 0] = 1
    from skimage.morphology import label
    from collections import OrderedDict
    import skimage
    kernel = skimage.morphology.ball(2)
    label_np = skimage.morphology.erosion(label_np, kernel)
    region_volume = OrderedDict()
    label_map, numregions = label(label_np == 1, return_num=True)
    region_volume['num_region'] = numregions
    max_region = 0
    total_volume = 0
    max_region_flag = 0
    print("region num :", numregions)
    for l in range(1, numregions + 1):
        region_volume[l] = np.sum(label_map == l)  # * volume_per_volume
        if region_volume[l] > max_region:
            max_region = region_volume[l]
            max_region_flag = l
        total_volume += region_volume[l]
        print("region {0} volume is {1}".format(l, region_volume[l]))
    post_label_np = label_np.copy()
    post_label_np[label_map != max_region_flag] = 0
    post_label_np[label_map == max_region_flag] = 1
    kernel = skimage.morphology.ball(5)
    img_dialtion = skimage.morphology.dilation(post_label_np, kernel)
    return img_dialtion


if __name__ == '__main__':
    # ori_path, gt_path = '/home/ljc/code/AMOS22/data/AMOS22/imagesTr/amos_0001.nii.gz', '/home/ljc/code/AMOS22/data/AMOS22/labelsTr/amos_0001.nii.gz'
    # cut_(ori_path, gt_path, '/home/ljc/code/AMOS22/data/AMOS22/')
    ori_root_path = '/home/ljc/code/AMOS22/data/AMOS22/imagesTr'
    gt_root_path = '/home/ljc/code/AMOS22/data/AMOS22/labelsTr'
    save_path = '/home/ljc/code/AMOS22/data/AMOS22/'
    ori_paths = os.listdir(ori_root_path)
    gt_paths = os.listdir(gt_root_path)
    ori_paths.sort()
    gt_paths.sort()
    for ori_path, gt_path in zip(ori_paths, gt_paths):
        cut_(os.path.join(ori_root_path, ori_path),
             os.path.join(gt_root_path, gt_path), save_path)
