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
import src.process.task2_sliding_window as task2_sliding_window
import src.process.task2_sliding_window2 as loader
import einops


def sliding_3D_window2(image, window_size, step):
    # image.shape = b, c, d, w, h
    # step must smaller than window_size
    image = rearrange(image, ' b c d w h -> b c w h d')
    _, _, width, height, depth = image.shape
    x_step, y_step, z_step = step
    x_window_size, y_window_size, z_window_size = window_size
    is_continue = False
    for z in range(0, depth, z_step):
        for y in range(0, height - y_window_size + 1, y_step):
            for x in range(0, width - x_window_size + 1, x_step):
                # 1 if z+z_window_size > depth
                # 2 if z.next() < depth -> z + step_z < depth
                # 3 step < window
                if z + z_window_size > depth and z - z_step + z_window_size != depth:
                    window = image[..., x:x_window_size + x, y:y_window_size + y, -1 - z_window_size:-1]
                else:
                    window = image[..., x:(x_window_size + x) if width > x_window_size else -1,
                             y:(y_window_size + y) if height > y_window_size else -1, z:z_window_size + z]
                # window = pad(window,)
                # _, _, w, h, d = window.shape
                # pad_w =
                # if window.shape[-1] == 16:
                #     print(window.shape)
                yield window


def sliding_3D_window3(image, window_height, step):
    # image.shape = b, c, d, w, h
    # step must smaller than window_size
    image = rearrange(image, ' b c d w h -> b c w h d')
    _, _, width, height, depth = image.shape
    # x_step, y_step, z_step = step
    # step only for z axis
    # x_window_size, y_window_size, z_window_size = window_size
    for z in range(0, depth, step):
        if window_height > depth:
            # padding
            gap = window_height - depth
            top_pad = gap // 2
            button_pad = gap - gap // 2
            window = torch.nn.functional.pad(image, (top_pad, button_pad), mode='constant', value=0)
        elif z + window_height > depth:
            window = image[..., -1 - window_height:-1]
        else:
            window = image[..., z:z + window_height]
        yield window


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
            v_acc.append(np.mean(
                calculate_acc(torch.argmax(output, dim=1), torch.argmax(target, dim=1), class_num=16, fun=DICE,
                              is_training=True)))
            v_loss.append(loss.item())
            print('\r \t {} / {}:valid_loss = {}'.format(index + 1, len(data_loader), loss.item()), end="")
    # ----------------------------------------------------
    # 返回每一个epoch的mean_loss
    print()
    return np.mean(t_loss), np.mean(v_loss), np.mean(v_acc)


def train_and_valid_model_slidingwindow(epoch, model, data_loader, device, optimizer, criterion):
    # ---------------------------------------------------
    t_loss = []
    v_loss = []
    v_acc = []
    # ---------------------------------------------------
    train_loader, valid_loader = data_loader
    model.train()
    model.to(device)
    for index, (x, y) in enumerate(train_loader):
        y = torch.LongTensor(y.long())
        y = torch.nn.functional.one_hot(y, 16)
        y = einops.rearrange(y, 'b d w h c -> b c d w h')
        fun = sliding_3D_window3
        for x_win, y_win in zip(fun(x, window_height=64, step=48),
                                fun(y, window_height=64, step=48)):
            # x_batch = x[..., x_win[2]:x_win[3], x_win[4]:x_win[5], x_win[0]:x_win[1]]
            # y_batch = y[..., y_win[2]:y_win[3], y_win[4]:y_win[5], y_win[0]:y_win[1]]
            optimizer.zero_grad()
            x_batch, y_batch = x_win, y_win
            if x_batch.shape != (1, 1, 256, 256, 64):
                continue
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # print(x_batch.shape)
            # print(y_batch.shape)
            pred = model(x_batch)
            loss = criterion(pred, y_batch.float())
            loss.backward()
            optimizer.step()
            t_loss.append(loss.item())
            print('\r \t {} / {}:train_loss = {}'.format(index + 1, len(train_loader), loss.item()), end="")
    model.eval()
    for index, (data, y) in enumerate(valid_loader):
        # valid data

        model.cpu()
        y = torch.LongTensor(y.long())
        y = one_hot(y, 16)
        target = rearrange(y, 'b d w h c -> b c d w h')
        # training param
        output = model(data.float())
        loss = criterion(output, target.float())
        v_acc.append(np.mean(
            calculate_acc(torch.argmax(output, dim=1), torch.argmax(target, dim=1), class_num=16, fun=DICE,
                          is_training=True)))
        v_loss.append(loss.item())
        print('\r \t {} / {}:valid_loss = {}'.format(index + 1, len(valid_loader), loss.item()), end="")
    # ----------------------------------------------------
    # 返回每一个epoch的mean_loss
    print()
    return np.mean(t_loss), np.mean(v_loss), np.mean(v_acc)


def train(pre_train_model, n_epochs, batch_size, optimizer, criterion, device, is_load):
    path = os.path.join('..', 'checkpoints', 'auto_save_task2_sliding_window')
    train_loader = loader.get_dataloader(batch_size=batch_size)
    valid_loader = loader.get_valid_data()
    train_valid_loader = [train_loader, valid_loader]
    train_loss = []
    valid_loss = []
    valid_acc = []
    # ----------------------------------------------------------------
    for epoch in range(1, n_epochs + 1):
        print('{} / {} epoch:'.format(epoch, n_epochs))
        t_loss, v_loss, v_acc = train_and_valid_model_slidingwindow(epoch=epoch, model=pre_train_model,
                                                                    data_loader=train_valid_loader,
                                                                    device=device, optimizer=optimizer,
                                                                    criterion=criterion)
        # 每20次保存一次模型
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
    print('beginning training')
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
