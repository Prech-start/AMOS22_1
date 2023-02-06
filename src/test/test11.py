import SimpleITK as sitk
from skimage.segmentation import slic
import numpy as np

i = sitk.ReadImage('/home/ljc/code/AMOS22/data/AMOS22/tr_ori/amos_0001.nii.gz')
im = sitk.GetArrayFromImage(i)
img = np.expand_dims(im, 0)
img = np.clip(img, a_min=-175, a_max=250)
img = (img + 175) / 425 * 255
segmentation = slic(img, n_segments=1000, compactness=0.1, channel_axis=0)
pass
segmentation = segmentation.astype(np.int32)
pic = sitk.GetImageFromArray(segmentation)
pic.SetSpacing(i.GetSpacing())
sitk.WriteImage(pic, '/home/ljc/Desktop/slic.nii.gz')
pass