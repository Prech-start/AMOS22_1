import numpy as np
import os
import json
import SimpleITK as sitk
import pandas as pd


def get_info_():
    json_path = os.path.join('..', '..', 'data', 'AMOS22', 'task1_dataset.json')
    task_json = json.load(open(json_path))
    train_paths = task_json['training']
    dicts = [
        'image_name',
        'image_min',
        'image_max',
        'image_mean',
        'image_median',
        'image_size',
        'label_name',
        'label_num_class'
    ]
    info_xlsx = []
    for image_label in train_paths:
        image_path = image_label['image']
        label_path = image_label['label']
        image_array = load_(image_path)
        label_array = load_(label_path)
        image_min = np.min(image_array)
        image_max = np.max(image_array)
        image_mean = np.mean(image_array)
        image_median = np.median(image_array)
        image_size = image_array.shape
        label_num_class = np.unique(label_array)
        info_xlsx.append(
            [image_path, image_min, image_max, image_mean
                , image_median, image_size, label_path, label_num_class])
    info_xlsx = np.array(info_xlsx)
    info_xlsx = pd.DataFrame(info_xlsx)
    info_xlsx.columns = dicts
    info_xlsx.to_csv(os.path.join('..', 'output', 'info_task1.csv'))


def load_(path):
    image_ = sitk.ReadImage(os.path.join('..', '..', 'data', 'AMOS22', path))
    return np.array(sitk.GetArrayFromImage(image_)).astype(np.float32)


get_info_()
