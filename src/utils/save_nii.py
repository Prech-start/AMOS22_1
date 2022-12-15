import SimpleITK as sitk
import numpy as np


def np2nii(array, spacing, outDir=None):
    '''
    save numpy array as simple itk 
    inputs:: 
        array: np.array  int type
        spacing: can get from the image.GetSpacing()    ##image = sitk.ReadImage(image_file)
        outDir: save directory
    output:: 
        array_sitk
    '''
    array_sitk = sitk.GetImageFromArray(array)
    array_sitk.SetSpacing(spacing)  ##SetSize
    # image_np_crop_sitk.SetOutputDirection(image.GetDirection()) 
    if outDir is not None:
        sitk.WriteImage(array_sitk, outDir)
    return array_sitk


def saveNii(img_file, label_np, ):
    '''
    img_file:: path of input nii file
    label_np:: prediction  np.array  int type
    '''
    img_itk = sitk.ReadImage(img_file)
    label_itk = sitk.GetImageFromArray(label_np)
    label_itk.CopyInformation(img_itk)
    output_dir = img_file.replace('_img.', '_label.')
    sitk.WriteImage(label_itk, output_dir)
    print('save in ', output_dir)


def resample_image_v2(itk_image, out_spacing=[1.5, 1.5, 2.0], is_label=False):
    '''
    resize using simple itk 
    inputs:: 
        itk_image: itk object   e.g. image = sitk.ReadImage(img_path)
        out_spacing: can get from the image.GetSpacing()    ##image = sitk.ReadImage(image_file)
        outDir: save directory
    output:: 
        array_sitk
    '''
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, out_spacing)]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:  # 如果是mask图像，就选择sitkNearestNeighbor这种插值
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:  # 如果是CT/MRI图像，就采用sitkBSpline插值法
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)
