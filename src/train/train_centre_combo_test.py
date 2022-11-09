import sys
from einops import *

from src.utils.image_process import save_image_information
from src.utils.train_utils import *

from src.train.loss import *

sys.path.append('..')
from src.utils.accuracy import *

import src.process.task2_data_loader_centre as loader
from src.model.model_test import *


def train_and_valid_model(epoch, model, data_loader, device, optimizer, criterion):
    n_epoch, max_epoch = epoch
    # ---------------------------------------------------
    t_loss = []
    v_loss = []
    v_acc = []
    train_loader, valid_loader = data_loader
    loss_dice, loss_L1 = criterion
    weight = ((n_epoch - max_epoch) ** 2) / ((1 - max_epoch) ** 2)
    # ---------------------------------------------------
    model.train()
    model.to(device)
    criterion.to(device)
    for index, (data, GT, GT_centre) in enumerate(train_loader):
        # train_data
        optimizer.zero_grad()
        # trans GT to onehot
        GT = torch.LongTensor(GT.long())
        data = data.float().to(device)
        GT = one_hot(GT, 16)
        GT = rearrange(GT, 'b d w h c -> b c d w h')
        # training param
        output, out_centre = model(Variable(data))
        GT = GT.to(device)
        loss_ = loss_dice(output, GT.float())
        GT_centre = GT_centre.unsqueeze(0)
        GT_centre = GT_centre.to(device)
        # print(out_centre.shape, GT_centre.shape)
        loss_centre = loss_L1(out_centre, GT_centre.float())
        loss = (1 - weight) * loss_ + weight * loss_centre
        loss.backward()
        optimizer.step()
        t_loss.append(loss.item())
        print('\r \t {} / {}:train_loss = {}'.format(index + 1, len(train_loader), loss.item()), end="")
    print()
    model.eval()
    model.cpu()
    criterion.cpu()
    for index, (data, GT, GT_centre) in enumerate(valid_loader):
        # valid data
        GT = torch.LongTensor(GT.long())
        GT = one_hot(GT, 16)

        target = rearrange(GT, 'b d w h c -> b c d w h')
        # training param
        output, _ = model(data.float())
        loss = loss_dice(output, target.float())
        v_acc.append(
            calculate_acc(output, target, class_num=16, fun=DICE,
                          is_training=True))
        v_loss.append(loss.item())
        print('\r \t {} / {}:valid_loss = {}'.format(index + 1, len(valid_loader), loss.item()), end="")
    # ----------------------------------------------------
    # 返回每一个epoch的mean_loss
    print()
    return np.mean(t_loss), np.mean(v_loss), np.mean(v_acc)


def train(pre_train_model, n_epochs, batch_size, optimizer, criterion, device, is_load, strategy):
    path_dir = os.path.dirname(__file__)
    path = os.path.join(path_dir, '..', 'checkpoints', strategy)
    if not os.path.exists(path):
        os.makedirs(path)
    train_loader = loader.get_train_data(batch_size=batch_size)
    valid_loader = loader.get_valid_data()
    train_valid_loader = [train_loader, valid_loader]
    if is_load:
        with open(os.path.join('{}.tmp'.format(strategy)), 'rb+') as f:
            train_loss, valid_loss, valid_acc = pickle.load(f)
            max_acc = 0.5
    else:
        train_loss = []
        valid_loss = []
        valid_acc = []
        max_acc = 0.
    import datetime
    start = time.time()
    # ----------------------------------------------------------------
    for epoch in range(1, n_epochs + 1):
        seconds = time.time() - start
        print('{} / {} epoch, cost_time {:.0f}min:'.format(epoch, n_epochs, seconds // 60))
        t_loss, v_loss, v_acc = train_and_valid_model(epoch=[epoch, n_epochs], model=pre_train_model,
                                                      data_loader=train_valid_loader,
                                                      device=device, optimizer=optimizer,
                                                      criterion=criterion)
        # 每次保存最新的模型
        torch.save(pre_train_model.state_dict(), os.path.join(path, 'Unet-new.pth'))
        # 保存最好的模型
        if v_acc > max_acc:
            torch.save(pre_train_model.state_dict(), os.path.join(path, 'Unet-final.pth'))
            max_acc = v_acc
        train_loss.append(t_loss)
        valid_loss.append(v_loss)
        valid_acc.append(v_acc)
        # 保存训练的loss
        save_loss(train_loss, valid_loss, valid_acc, strategy)
        pic_loss_acc(strategy)
    return pre_train_model



def run():
    print('beginning training')
    class_num = 16
    learning_rate = 1e-4
    max_epoch = 300
    model = UnetModel_centre(1, class_num, 6)
    strategy = 'centre+combo'
    # 是否加载模型
    is_load = False
    # 是否迁移模型
    is_move = False
    if is_load:
        model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', strategy, 'Unet-new.pth')))
    if is_move:
        model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', strategy, 'Unet-final.pth')))
    loss_weight = [1, 2, 2, 3, 6, 6, 1, 4, 3, 4, 7, 8, 10, 5, 4, 5]
    loss = ComboLoss2(loss_weight)
    loss_centre = torch.nn.L1Loss()
    # 150 768min
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = train(pre_train_model=model, n_epochs=max_epoch, batch_size=1, optimizer=optimizer,
                  criterion=[loss, loss_centre],
                  device=torch.device('cpu'), is_load=is_load, strategy=strategy)

if __name__ == '__main__':
    run()