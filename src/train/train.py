import os
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


def train_and_valid_model(epoch, model, data_loader, device, optimizer, criterion):
    # ---------------------------------------------------
    t_loss = []
    v_loss = []
    # ---------------------------------------------------
    for index, (data, y) in enumerate(data_loader):
        if index < len(data_loader) - 2:
            # train_data
            model.train()
            model.to(device)
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
            t_loss.append(loss.item())
            print('\r \t {} / {}:train_loss = {}'.format(index + 1, len(data_loader), loss.item()), end="")
        else:
            # valid data
            model.eval()
            model.cpu()
            y = torch.LongTensor(y.long())
            y = one_hot(y, 16)
            target = rearrange(y, 'b d w h c -> b c d w h')
            # training param
            output = model(data.float())
            loss = criterion(output, target.float())
            v_loss.append(loss.item())
            print('\r \t {} / {}:valid_loss = {}'.format(index + 1, len(data_loader), loss.item()), end="")
    # ----------------------------------------------------
    # 返回每一个epoch的mean_loss
    print()
    return np.mean(t_loss), np.mean(v_loss)


def train(pre_train_model, batch_size, optimizer, criterion, device):
    path = os.path.join('..', 'checkpoints', 'auto_save')
    n_epochs = 300
    train_valid_loader = get_dataloader(batch_size=batch_size)
    train_loss = []
    valid_loss = []
    # ----------------------------------------------------------------
    for epoch in range(1, n_epochs + 1):
        print('{} / {} epoch:'.format(epoch, n_epochs))
        t_loss, v_loss = train_and_valid_model(epoch=epoch, model=pre_train_model,
                                                       data_loader=train_valid_loader,
                                                       device=device, optimizer=optimizer, criterion=criterion)
        # 每30次保存一次模型
        if epoch % 30 == 0:
            torch.save(pre_train_model.state_dict(), os.path.join(path, 'Unet-{}.pth'.format(epoch)))
        torch.save(pre_train_model.state_dict(), os.path.join(path, 'Unet-final.pth'))
        train_loss.append(t_loss)
        valid_loss.append(t_loss)
        # 保存训练的loss
        save_loss(train_loss, valid_loss)
    pic_loss_line()
    return pre_train_model


def show_result(model):
    # 获取所有的valid样本
    test_data = data_set(False)
    data_loader = DataLoader(
        dataset=test_data,
        batch_size=1,
        pin_memory=True,
        shuffle=True
    )
    with torch.no_grad():
        # 对每一个测试案例做展示并保存
        for index, (x, y) in enumerate(data_loader):
            PRED = model(x.float())
            result = torch.argmax(PRED, dim=1)
            result = result.data.squeeze().cpu().numpy()
            save_image_information(index, result)
            pass

if __name__ == '__main__':
    class_num = 16
    learning_rate = 1e-4
    model = UnetModel(1, class_num, 6)
    # model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'Generalized_Dice_loss_e-3_1.pth')))
    # loss = Generalized_Dice_loss([1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4])
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # TODO: loss = nn.BCELoss() -
    model = train(pre_train_model=model, batch_size=1, optimizer=optimizer, criterion=loss, device=torch.device('cuda'))