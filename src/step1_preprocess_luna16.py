import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import numpy
import pandas
import ntpath
import cv2  # conda install -c https://conda.anaconda.org/menpo opencv3
import shutil
import random
import math
# from bs4 import BeautifulSoup #  conda install beautifulsoup4, coda install lxml
import os
import glob
import sys
import scipy.misc
import scipy.ndimage as ndi
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
import matplotlib as plt
from skimage.morphology import ball, disk, binary_closing,binary_erosion,binary_dilation
from skimage.segmentation import clear_border
from skimage.measure import label,regionprops, perimeter
from skimage.filters import roberts, sobel
import scipy.signal as signal
from PIL import Image


random.seed(1321)
numpy.random.seed(1321)

import numpy as np
import pydicom as dicom
def load_scans_normalization(path_path, pt_no=None):
    # path = path_path + '/img/{}_0000.nii.gz'.format(pt_no)
    data_path_ =path_path[pt_no] + r'/*.dcm'
    data_addrs = glob.glob(data_path_)
    data_addrs = sorted(data_addrs, key=lambda s: s.lower())
    slices = [dicom.read_file(s, force=True) for s in data_addrs]
    slices_ = [s.pixel_array for s in slices]
    slices = np.stack(slices, axis=0)
    print(data_path_)
    print(slices.shape)
    return slices


def save_lung_image(data_path, dst_dir=r'./'):

    data=glob.glob(data_path+"*.png")
    for i in range(len(data)):
        img = data[i]
        img = normalize(img)
        img=get_lung_parenchyma(img_temp=img) 
        cv2.imwrite(dst_dir+ "img_" + str(i).rjust(4, '0') + "_"+str(thickness)+"mm_i.png", img*255) 
    data=[]

def normalize(image):

    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


import heapq
import skimage
from skimage import measure 
def keep_connected_area(img_mat_axial, pt_no=None):

    connected_areas = measure.label(img_mat_axial, neighbors=8)
    print(np.max(connected_areas), np.min(connected_areas))
    connected_areas_properties = measure.regionprops(connected_areas)

    ## find the index of the first 1 largest number in the label
    list_areas = [prop.area for prop in connected_areas_properties]
    max_num_index_list = map(list_areas.index, heapq.nlargest(1, list_areas))
    max_num_index_list = list(max_num_index_list)
    print(connected_areas_properties[max_num_index_list[0]].label)

    ## mask all the elements of the index with 1
    first_largest = connected_areas_properties[max_num_index_list[0]].label
    # second_largest = connected_areas_properties[max_num_index_list[1]].label
    connected_areas_1st = np.zeros_like(connected_areas)
    connected_areas_1st[connected_areas == first_largest] = 1
    # connected_areas_1st[connected_areas == second_largest] = 1
    # connected_areas_2st = connected_areas[connected_areas==second_largest]
    return connected_areas_1st


def get_lung_parenchyma(img_temp):
    img=img_temp.astype(numpy.float64)

    middle=img
    mean = numpy.mean(middle[middle>0])  ##print mean value of the normalize CT slice   mean = 0.58
    
    count=0
    total_value=0
    for i in range(middle.shape[0]):
        for j in range(middle.shape[1]):
            if middle[i,j]>0:
                total_value+=middle[i,j]
                count+=1
    
    mean= total_value/count   ##print mean value of the normalize CT slice   mean = 0.58788

    img_binary = img < mean

    cleared = clear_border(img_binary)  ##clear the border of the image
    cleared = morphology.remove_small_objects(cleared,min_size=500,connectivity=1)

    label_image = label(cleared)  ##determine the number and labels of connected regions in an image  ##https://blog.csdn.net/zz2230633069/article/details/85107971  ##https://www.cnblogs.com/yqpy/p/14440679.html
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0

    binary = morphology.remove_small_objects(binary,min_size=1500,connectivity=1) # one last dilation
    img_1 = binary
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    img_2 = binary
    selem = disk(20)
    binary = binary_closing(binary, selem)
    img_3 = binary
    edges = roberts(binary)
    img_4 = edges
    binary = ndi.binary_fill_holes(edges)
    img_5 = binary  ##LungParenchyma
    binary = morphology.dilation(binary,numpy.ones([8,8])) # one last dilation
    img_6 = binary
    img[binary==0]=0
    img_7 = img


    img_5 = binary_erosion(img_5, disk(2))
    img[img_5==0]=0
    img_binary_Lesion = img > mean
    try:
        img_binary_Lesion = keep_connected_area(img_binary_Lesion)   ###keep the largest connected area
    except Exception as e:
        pass
    
    # [img>mean]
    # img_8 = cv2.addweighted(img_temp, 1, img_binary_Lesion*1.0, 1, 0.8)
    # img_8 = Image.blend(img_temp*255.0, img_binary_Lesion*255.0, 1)
    img_binary_Lesion = img_binary_Lesion*1   ## don't multiple 255
    img_binary_Lesion_ = img_binary_Lesion*255   ## don't multiple 255
    img_binary_Lesion_ = Image.fromarray(img_binary_Lesion_.astype('uint8')).convert('RGB')   ## don't multiple 255
    img_binary_Lesion = Image.fromarray(img_binary_Lesion.astype('uint8'), 'P')
    palettedata = [0, 0, 0, 102, 0, 255, 0, 255, 176]
    img_binary_Lesion.putpalette(palettedata)
    img_binary_Lesion = img_binary_Lesion.convert('RGB')
    # image = pil_image.fromarray(image[:, :, 0].astype('uint8'), 'L')
    img_ori = img_temp*255.0
    img_ori = Image.fromarray(img_ori.astype('uint8'), 'L')
    img_ori = img_ori.convert('RGB')
    img_8 = Image.blend(img_ori, img_binary_Lesion, 0.7) # blend_img = img1 * (1 – 0.3) + img2* alpha

    # return img
    # return img, img_binary*1, cleared*1, label_image*1, img_1*1, img_2*1, img_3*1, img_4*1, img_5*1, img_6*1, img_7*1, img_binary_Lesion*1, img_8
    return img_binary_Lesion_, img_8

path_path = [r'./data']
# slices = load_scans_normalization(path_path, pt_no=0)
s = r'.\data\IMG-0001-00025.dcm'

# slices = dicom.read_file(s, force=True).pixel_array
data_path_ =path_path[0] + r'/*.dcm'
data_addrs = glob.glob(data_path_)
data_addrs = sorted(data_addrs, key=lambda s: s.lower())

for i in range(len(data_addrs)):
    file = SimpleITK.ReadImage(data_addrs[i])   ##s
    data = SimpleITK.GetArrayFromImage(file)
    print(np.max(data), np.min(data))
    # data = data+1024.0
    data = data[0]
    img = normalize(data)
    img_ori = img
    img = get_lung_parenchyma(img_temp=img) 
    thickness = 5
    img[0].save('./segm_parenchyma/im_{}.png'.format(i)) # 注意jpg和png，否则 OSError: cannot write mode RGBA as JPEG
    img[1].save('./segm_parenchyma/im1_{}.png'.format(i)) # 注意jpg和png，否则 OSError: cannot write mode RGBA as JPEG

    # dst_dir = r'./segm/'
    # cv2.imwrite(dst_dir+ "original_" + str(i).rjust(4, '0') + "_"+str(thickness)+"mm_i.png", img_ori*255) 
    # cv2.imwrite(dst_dir+ "img_" + str(i).rjust(4, '0') + "_"+str(thickness)+"mm_i.png", img[0]*255) 
    # # if i==11:
    # for im_i in range(1,len(img)):
    #     cv2.imwrite(dst_dir+ "img_" + str(i).rjust(4, '0') + "_"+str(thickness)+"mm_"+str(im_i)+".png", img[im_i]*255) 
    # # dst_dir = r'./segm_parenchyma/'
    # # for im_i in range(len(img)-2,len(img)):
    # #     cv2.imwrite(dst_dir+ "img_" + str(i).rjust(4, '0') + "_"+str(thickness)+"mm_"+str(im_i)+".png", img[im_i]*255) 


print()
