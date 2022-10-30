
from wama.utils import *

img_path = r"D:\git\testnini\s1_v1.nii"
mask_path = r"D:\git\testnini\s1_v1_m1_w.nii"

subject1 = wama()  # build instance
subject1.appendImageFromNifti('CT', img_path)  # Load image, custom modal name, such as 'CT'
subject1.appendSementicMaskFromNifti('CT', mask_path)  # Load the mask, pay attention to the corresponding modal name
# also can use appendImageAndSementicMaskFromNifti to load both image and mask at the same time

subject1.resample('CT', aim_spacing=[1, 1, 1])  # Resample to 1x1x1 mm (note the unit is mm)
subject1.adjst_Window('CT', WW = 321, WL = 123) # Adjust window width and window level

# 3D visualization
subject1.show_scan('CT', show_type='slice')  # Display original image in slice mode
subject1.show_scan('CT', show_type='volume')  # Display original image in volume mode
subject1.show_MaskAndScan('CT', show_type='volume') # Display original image and mask at the same time
subject1.show_bbox('CT', 2)  # Display the bbox shape. Note that when there is no bbox, the minimum external matrix is automatically generated from the mask as bbox
