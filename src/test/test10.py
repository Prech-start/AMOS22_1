import os
import shutil
import json

path_dir = os.path.dirname(__file__)

task2_json = json.load(open(os.path.join(path_dir, '..', '..', 'data', 'AMOS22', 'task2_dataset.json')))

file_path = [[os.path.join('/home/ljc/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task01_AMOSS22', path_['image']),
              os.path.join('/home/ljc/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task01_AMOSS22', path_['label'])]
             for path_ in task2_json['training']]

CT_train_path = file_path[0:150]
CT_valid_path = file_path[150:160]
CT_test_path = file_path[160:200]
MRI_train_path = file_path[200:230]
MRI_valid_path = file_path[230:232]
MRI_test_path = file_path[232::]
train_path = CT_train_path + MRI_train_path
valid_path = CT_valid_path + MRI_valid_path
test_path = CT_test_path + MRI_test_path


def move(old_path, new_path):
    shutil.move(old_path, new_path)


# 该功能可实现nnUNet的Json文件生成


from batchgenerators.utilities.file_and_folder_operations import save_json, subfiles
from typing import Tuple
import numpy as np

'''
	获取文件夹内独立文件 【列表】
'''


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-7] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques


def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, license: str = "CC-BY-SA 4.0", dataset_description: str = "",
                          dataset_reference="SRIDB x CUHKSZ x HKU x SYSU x LGCHSZ x LGPHSZ",
                          dataset_release='1.0 01/05/2022'):
    # 获取文件夹内各个独立的文件
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)
    # imagesTs_dir 文件夹可以为空，只要有训练的就行
    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = "AMOS22"
    json_dict['description'] = "MICCAI2022 Multi-Modality Abdominal Multi-Organ Segmentation Task 2"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {"0": "CT"}
    json_dict['labels'] = {
        "0": "background",
        "1": "spleen",
        "2": "right kidney",
        "3": "left kidney",
        "4": "gall bladder",
        "5": "esophagus",
        "6": "liver",
        "7": "stomach",
        "8": "arota",
        "9": "postcava",
        "10": "pancreas",
        "11": "right adrenal gland",
        "12": "left adrenal gland",
        "13": "duodenum",
        "14": "bladder",
        "15": "prostate/uterus"

    }

    # 下面这些内容不需要查看和更改
    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
        in
        train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    output_file += "dataset.json"
    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file))


if __name__ == "__main__":
    # 自行修改文件路径，当前在windows环境下操作
    output_file = r'/home/ljc/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task01_AMOSS22/'
    imagesTr_dir = r'/home/ljc/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task01_AMOSS22/imagesTr'
    imagesTs_dir = r'/home/ljc/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task01_AMOSS22/imagesTs'
    labelsTr = r'/home/ljc/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task01_AMOSS22/labelsTr'

    # 只需要给出空定义，具体内容在上面的函数中修改
    modalities = ''
    labels = {

    }
    get_identifiers_from_splitted_files(output_file)
    generate_dataset_json(output_file,
                          imagesTr_dir,
                          imagesTs_dir,
                          labelsTr,
                          modalities,
                          labels
                          )
