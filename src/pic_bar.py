from visdom import Visdom
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

if __name__ == '__main__':
    # path = '/home/ljc/code/AMOS22/data/AMOS22/imagesTr/amos_0001.nii.gz' # AMOS
    # path = '/media/ljc/ugreen/dataset/Abdomen/RawData/Training/img/img0004.nii.gz' # BTCV
    path = '/media/ljc/ugreen/dataset/WORD/WORD-V0.1.0/imagesTr/word_0003.nii.gz' # WORD
    x = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.int16)
    x = x.flatten()
    x = np.clip(x, -175, 1500)
    x.sort()
    array = np.where(x > -175)
    x = x[array[0][0]:-1]
    # x = x.reshape(-1, 1)
    plt.hist(x, bins=100)
    plt.show()
    pass
