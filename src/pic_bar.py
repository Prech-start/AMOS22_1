from visdom import Visdom
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

if __name__ == '__main__':
    path = '/media/ljc/ugreen/dataset/Abdomen/RawData/Training/img/img0002.nii.gz'
    x = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.int16)
    x = x.flatten()
    x = np.clip(x, 0, 99999)
    x.sort()
    array = np.where(x > 0)
    x = x[array[0][0]:-1]
    # x = x.reshape(-1, 1)
    plt.hist(x, bins=100)
    plt.show()
    pass
