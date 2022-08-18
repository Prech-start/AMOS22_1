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
from loss import BCELoss_with_weight
sys.path.append('..')
from src.utils.accuracy import *
import src.process.task2_data_loader as task2_data_loader

def train_and_valid_model(epoch, model, data_loader, device, optimizer, criterion):
    # ---------------------------------------------------
    t_loss = []
    v_loss = []
    v_acc = []
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
            v_acc.append(np.mean(calculate_acc(torch.argmax(output, dim=1), torch.argmax(target, dim=1), class_num=16, fun=DICE, is_training=True)))
            v_loss.append(loss.item())
            print('\r \t {} / {}:valid_loss = {}'.format(index + 1, len(data_loader), loss.item()), end="")
    # ----------------------------------------------------
    # 返回每一个epoch的mean_loss
    print()
    return np.mean(t_loss), np.mean(v_loss), np.mean(v_acc)


def train(pre_train_model, n_epochs, batch_size, optimizer, criterion, device, is_load):
    path = os.path.join('..', 'checkpoints', 'auto_save_task2')
    train_valid_loader = task2_data_loader.get_dataloader(batch_size=batch_size)
    train_loss = []
    valid_loss = []
    valid_acc = []
    # ----------------------------------------------------------------
    for epoch in range(1, n_epochs + 1):
        print('{} / {} epoch:'.format(epoch, n_epochs))
        t_loss, v_loss, v_acc = train_and_valid_model(epoch=epoch, model=pre_train_model,
                                               data_loader=train_valid_loader,
                                               device=device, optimizer=optimizer, criterion=criterion)
        # 每30次保存一次模型
        if epoch % 20 == 0:
            torch.save(pre_train_model.state_dict(), os.path.join(path, 'Unet-{}.pth'.format(epoch)))
        torch.save(pre_train_model.state_dict(), os.path.join(path, 'Unet-final.pth'))
        train_loss.append(t_loss)
        valid_loss.append(v_loss)
        valid_acc.append(v_acc)
        # 保存训练的loss
        if not is_load:
            save_loss(train_loss, valid_loss, valid_acc)
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
    epoch = 300
    model = UnetModel(1, class_num, 6)
    # 是否加载模型
    is_load = False
    # 是否迁移模型
    is_move = False
    if is_load:
        model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'Unet-180.pth')))
    if is_move:
        model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'Unet-210.pth')))
    loss_weight = [1, 2, 2, 3, 6, 6, 1, 4, 3, 4, 7, 8, 10, 5, 4, 5]
    loss = BCELoss_with_weight(loss_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # TODO: loss = nn.BCELoss()
    model = train(pre_train_model=model, n_epochs=epoch, batch_size=1, optimizer=optimizer, criterion=loss,
                  device=torch.device('cuda'), is_load=is_load)
