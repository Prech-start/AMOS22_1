from src.model.model import *
from src.train.train_centre import train
import torch.nn as nn
import os, sys
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
    print('beginning training')
    class_num = 16
    learning_rate = 3e-5
    epoch = 300
    model = UnetModel_centre(1, class_num)
    # 是否加载模型
    is_load = False
    # 是否迁移模型
    is_move = False
    if is_load:
        model.load_state_dict(torch.load(os.path.join('src', 'checkpoints', 'auto_save_task2_centre', 'Unet-final.pth')))
    if is_move:
        model.load_state_dict(torch.load(os.path.join('src', 'checkpoints', 'auto_save', 'Unet-210.pth')))
    loss_weight = [1, 2, 2, 3, 6, 6, 1, 4, 3, 4, 7, 8, 10, 5, 4, 5]
    loss = BCELoss_with_weight(loss_weight)
    loss_centre = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = train(pre_train_model=model, n_epochs=epoch, batch_size=1, optimizer=optimizer,
                  criterion=[loss, loss_centre],
                  device=torch.device('cuda:0'), is_load=is_load)
