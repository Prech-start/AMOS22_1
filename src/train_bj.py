# coding: utf-8
import numpy as np
from scipy import io
# import dicom as dicom
import random
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image as pil_image
from torchvision.utils import save_image
import torch.nn.functional as F
import nibabel as nib
import glob


torch.autograd.set_detect_anomaly(True)
import SimpleITK as sitk
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.functional import one_hot
import matplotlib.pyplot as plt
from einops import rearrange
from dataset_bj import data_set, get_dataloader
from model.model import UnetModel

random.seed(2018)
torch.manual_seed(2018)


class DiceLoss(nn.Module):
    def __init__(self):  # weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        m1 = logits.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = (2.0 * intersection.sum(1) + 1.0) / (m1.sum(1) + m2.sum(1) + 1.0)
        score = 1 - score.sum() / num
        return score


root_path = r'/media/bj/DataFolder3/datasets/challenge_AMOS22'
dataset_name = 'AMOS22'
json_name = 'task1_dataset.json'
real_path = os.path.join(root_path, dataset_name, json_name)
task_json = json.load(open(real_path))
paths_train = task_json['training']
path_train = paths_train[0]['image']  ##'label' 
image_paths = [paths_train[i]['image'] for i in range(len(paths_train))]
label_paths = [paths_train[i]['label'] for i in range(len(paths_train))]
image_paths_full = [os.path.join(root_path, dataset_name, image_paths[i]) for i in range(len(image_paths))]
label_paths_full = [os.path.join(root_path, dataset_name, label_paths[i]) for i in range(len(label_paths))]

model = UnetModel(1, 16, 6)

load_model_continue = False  # False
if load_model_continue:
    print('load old model(weights) and continue training')
    if os.path.exists('./pth/Unet.pth'):
        model.load_state_dict(torch.load('./pth/Unet.pth', map_location=lambda storage, loc: storage))
    else:
        raise ValueError('did not find the model---Unet.pth')
epoches = 300
lr = 1e-4
criterion_dice = DiceLoss()
criterion = nn.BCELoss()
# criterion_dice(outputs, Variable(labels[i+in_features//2].type(Tensor)))
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def train_(model, data_loader, epoch, epoches, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.train()
    # parb = tqdm(data_loader)
    t_loss = []
    model = model.to(device)
    for batch, (data, y) in enumerate(data_loader):
        optimizer.zero_grad()
        # trans y to onehot
        y = torch.LongTensor(y.long())
        data, y = data.float().to(device), y.to(device)
        y = one_hot(y, 16)
        target = rearrange(y, 'b d w h c -> b c d w h')
        # training param
        output = model(Variable(data))
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()
        # parb.set_description('epoch:{}, loss:{}'.format(epoch, loss.item()))
        print(
            'Epoch {}/{},batch index: {}/{} loss: {:.7f}'.format(epoch, epoches, batch, len(data_loader), loss.item()))
        t_loss.append(loss.item())
    print("**************Epoch [%d] Avarage Loss: %.7f**************" % (epoch, np.mean(t_loss)))

    return t_loss


def validation(model, data_loader, epoch, epoches, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    v_loss = []
    model = model.to(device)
    with torch.no_grad():
        for test, y in data_loader:
            # model = model.cpu()
            y = torch.LongTensor(y.long())
            test, y = test.float().to(device), y.to(device)
            y = one_hot(y, 16)
            target = rearrange(y, 'b d w h c -> b c d w h')
            PRED = model(test.float())
            loss = criterion(PRED, target.float())
            v_loss.append(loss.item())
        print("****************Epoch [%d] Avarage Validation Loss: %.4f***************" % (epoch, np.mean(v_loss)))
    return v_loss


def draw_learning_curve(len_train, train_loss, valid_loss, save_dir='loss line.png'):
    plt.plot([i for i in range(len_train)], train_loss, '-', label='train_loss')
    plt.plot([i for i in range(len_train)], valid_loss, '-', label='valid_loss')
    plt.savefig(save_dir, bbox_inches='tight')


########################train

data_loader = get_dataloader(image_paths_full[:160], label_paths_full[:160], bs=1, ifshuffle=False)
test_loader = get_dataloader(image_paths_full[160:162], label_paths_full[160:162], bs=1, ifshuffle=False)
train_loss = []
validation_loss = []
for epoch in range(epoches * 0 + 1, epoches * 1 + 1):

    t_loss = train_(model, data_loader, epoch, epoches, device=torch.device('cuda'))
    v_loss = validation(model, test_loader, epoch, epoches, device=torch.device('cuda'))
    train_loss.append(np.mean(t_loss))
    validation_loss.append(np.mean(v_loss))
    if epoch % 30 == 0:
        torch.save(model.state_dict(), "./pth/Unet-%d.pth" % (epoch))
    torch.save(model.state_dict(), "./pth/Unet.pth")
    draw_learning_curve(len(train_loss), train_loss, validation_loss, save_dir='loss line.png')


# print()


## ##########  test


def test(model, image, y=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    test_loss = []
    model = model.to(device)
    with torch.no_grad():
        # model = model.cpu()
        image = image.float().to(device)
        PRED = model(image.float())
        if y != None:
            y = torch.LongTensor(y.long())
            y = y.to(device)
            y = one_hot(y, 16)
            target = rearrange(y, 'b d w h c -> b c d w h')
            loss = criterion(PRED, target.float())
            test_loss.append(loss.item())
            print("****************Avarage testing Loss: %.4f***************" % (loss.item()))
    return PRED


import skimage

image_paths_test = image_paths_full[162:170]
label_paths_test = label_paths_full[162:170]
device = torch.device("cuda")

model.load_state_dict(torch.load('./pth/Unet-270.pth', map_location=lambda storage, loc: storage))
for i in range(len(image_paths_test)):
    folder_name = image_paths_test[i].rsplit('/', 1)[1]
    image_sitk = sitk.ReadImage(image_paths_test[i])
    x = sitk.GetArrayFromImage(image_sitk)
    x = np.array(x, dtype=float)
    ori_img_size = x.shape
    ## normalization
    x = x + 1024.0
    x = np.clip(x, a_min=0, a_max=2048.0)
    x = x / 2048.0
    x = skimage.transform.resize(x, [96, 256, 256], order=1, preserve_range=True, anti_aliasing=False)

    x = torch.from_numpy(x).type(torch.FloatTensor)
    x = x.unsqueeze_(0)
    x = x.unsqueeze_(0)

    label_sitk = sitk.ReadImage(label_paths_test[i])
    y = sitk.GetArrayFromImage(label_sitk)
    y = np.array(y, dtype=int)
    y = skimage.transform.resize(y, [96, 256, 256], order=0, preserve_range=True, anti_aliasing=False)
    y = torch.from_numpy(y).type(torch.FloatTensor)
    y = y.unsqueeze_(0)

    pred = test(model, x, y, device=torch.device("cuda"))
    result = torch.argmax(pred, dim=1)
    result = result.data.squeeze().cpu().numpy()
    result = skimage.transform.resize(result, ori_img_size, order=0, preserve_range=True, anti_aliasing=False)
    [print(i_c, np.count_nonzero(result == i_c)) for i_c in range(0, 17)]
    result_sitk = sitk.GetImageFromArray(result)
    result_sitk.CopyInformation(label_sitk)
    sitk.WriteImage(result_sitk, './results/{}'.format(folder_name))

print()
