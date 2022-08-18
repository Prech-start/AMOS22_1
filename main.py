from src.model.model import *
from src.train.train import train
import torch.nn as nn
import os,sys
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
from matplotlib.pyplot import plot
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.process.data_load import *
from src.model.model import *
from einops import *
from tqdm import tqdm
from torch.nn.functional import one_hot
import copy
from src.utils.image_process import save_image_information
from src.utils.train_utils import *
import gc
from src.train.loss import BCELoss_with_weight
sys.path.append('..')
from src.utils.accuracy import *
import src.process.task2_data_loader as task2_data_loader
import einops
import torch

import src.model.model
from src.process.task2_sliding_window import get_dataloader, sliding_3D_window


if __name__ == '__main__':
    # class_num = 16
    # learning_rate = 1e-4
    # epoch = 300
    # model = UnetModel(1, class_num, 6)
    # # 是否加载模型
    # is_load = False
    # # 是否迁移模型
    # is_move = False
    # if is_load:
    #     model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'Unet-180.pth')))
    # if is_move:
    #     model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'Unet-210.pth')))
    # loss_weight = [1, 2, 2, 3, 6, 6, 1, 4, 3, 4, 7, 8, 10, 5, 4, 5]
    # loss = BCELoss_with_weight(loss_weight)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # # TODO: loss = nn.BCELoss()
    # model = train(pre_train_model=model, n_epochs=epoch, batch_size=1, optimizer=optimizer, criterion=loss,
    #               device=torch.device('cuda'), is_load=is_load)
    data_loader = get_dataloader()

    model = src.model.model.UnetModel(1, 16, 6)

    device = torch.device('cuda')
    model.to(device)
    for x, y in data_loader:
        y = torch.LongTensor(y.long())
        x, y = x.to(device), y.to(device)
        y = torch.nn.functional.one_hot(y, 16)
        y = einops.rearrange(y, 'b d w h c -> b c d w h')
        fun = sliding_3D_window
        for x_win, y_win in zip(fun(x, window_size=(32, 128, 128), step=(16, 64, 64)), fun(y, window_size=(32, 128, 128), step=(16, 64, 64))):
            print(x_win.shape)
            print(y_win.shape)


