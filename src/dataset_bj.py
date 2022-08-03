import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import pickle
import skimage
from skimage.transform import resize
import SimpleITK as sitk
import json
import os
from torchvision import transforms
import torch
from torch.utils.data import DataLoader


def load_json(dataset_name, json_name, root_path ):
    
    real_path = os.path.join(root_path, dataset_name, json_name)
    return json.load(open(real_path))

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image):
        return torch.from_numpy(image).type(torch.FloatTensor)

dicom_transform = transforms.Compose([ToTensor()])
label_transform = transforms.Compose([ToTensor()])

class data_set(Dataset):
    def __init__(self, image_paths, label_paths):
        ##
        self.image_paths = image_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index): 
        image_sitk = sitk.ReadImage( self.image_paths[index] )
        x = sitk.GetArrayFromImage(image_sitk) 
        label_sitk = sitk.ReadImage( self.label_paths[index] )
        y = sitk.GetArrayFromImage(label_sitk) 
        # print('image::', image_sitk.GetSpacing(), image_sitk.GetSize())
        # print('label::', label_sitk.GetSpacing(), label_sitk.GetSize())
        # print('x::', x.shape)
        # print('y::', y.shape)
        x = np.array(x, dtype=float) 
        y = np.array(y, dtype=int) 
        
        ## normalization
        x = x + 1024.0
        x = np.clip(x, a_min=0, a_max=2048.0)
        x = x / 2048.0 

        # x = np.expand_dims(x, 0)
        # # y = np.expand_dims(y, 0)
        # x = resize(x, (1, 96, 256, 256), preserve_range=True)
        # y = np.round(resize(y, (96, 256, 256), preserve_range=True))
        x = skimage.transform.resize(x, [96, 256, 256], order=1, preserve_range=True, anti_aliasing=False)
        y = skimage.transform.resize(y, [96, 256, 256], order=0, preserve_range=True, anti_aliasing=False)
        x = torch.from_numpy(x).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.FloatTensor)
        return x.unsqueeze_(0), y 

def get_dataloader(image_paths_full, label_paths_full, bs=1, ifshuffle=False):

    dataset = data_set(image_paths_full, label_paths_full) 

    dataloader = DataLoader(dataset=dataset,
                            batch_size=bs,
                            num_workers=4,
                            shuffle=ifshuffle)
    return dataloader