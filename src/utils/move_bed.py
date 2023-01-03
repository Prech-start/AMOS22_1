import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import SimpleITK as sitk

x = sitk.GetArrayFromImage(sitk.ReadImage('amos_0001.nii.gz'))

# img = Image.open(
#     "001.png")
# img.show()
img = np.array(x)
img = np.clip(img, a_min=-175, a_max=250)
img = (img + 175) / 425
plt.subplot(121)
plt.imshow(img[55, :, :], cmap='gray')
img_np = np.array(img)
label_np = np.zeros_like(img_np).astype(np.uint8)
label_np[img_np > 0] = 1
# plt.subplot(122)
# plt.imshow(label*255,cmap='gray')
# plt.show()


import SimpleITK as sitk
import numpy as np
from skimage.morphology import label
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *

# label_path = "/Users/cengwankang/Downloads/Kidney_Stone/data_origin/ZengGuiXing/segmentation_only_stone.mha"
# label_path = "/Users/cengwankang/Downloads/shenzheng/li-xiao-mei/segmentation_stone.nii.gz"
# label_image = sitk.ReadImage(label_path)
# label_np = sitk.GetArrayFromImage(label_image).astype(np.uint8)
# print("label space: ", label_image.GetSpacing())
# volume_per_volume = np.prod(label_image.GetSpacing())


region_volume = OrderedDict()
label_map, numregions = label(label_np == 1, return_num=True)
region_volume['num_region'] = numregions
total_volume = 0
print("region num :", numregions)
max_region = 0
max_region_flag = 0
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

print("total region volume is :", total_volume)

import skimage

# img = skimage.data.binary_blobs(100)
# skimage.io.imshow(img)
# skimage.io.show()

kernel = skimage.morphology.ball(5)
img_dialtion = skimage.morphology.dilation(post_label_np, kernel)


# for i in range(90):
plt.subplot(221)
plt.imshow(img[55, :, :], cmap='gray')
plt.subplot(222)
plt.imshow(label_np[55, :, :] * 255, cmap='gray')
plt.subplot(223)
plt.imshow(post_label_np[55, :, :] * 255, cmap='gray')
# plt.show()
plt.subplot(224)
plt.imshow(img_dialtion[55, :, :] * 255, cmap='gray')
plt.show()
pass

# img_dialtion = img_dialtion/255
new_img = img_dialtion * x
image = sitk.GetImageFromArray(new_img)
image.CopyInformation(sitk.ReadImage('amos_0001.nii.gz'))
sitk.WriteImage(image, 'amos_001.nii.gz')
pass