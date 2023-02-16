import torch
import torch.nn as nn
import os
import json
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import SimpleITK as sitk
import numpy as np
from skimage.transform import resize
import torch.nn.functional as F
from torch.nn.functional import one_hot
from einops import rearrange
import os.path
import pickle
import matplotlib
import numpy

matplotlib.use('agg')
import matplotlib.pyplot as plt
from visdom import Visdom
import time

######################################################
#################### DATALOADER ######################
# path_dir = os.path.dirname(__file__)
# task2_json = json.load(open(os.path.join(path_dir, '..', 'data', 'AMOS22', 'task2_dataset.json')))
# path_dir = r'../data/'
path_dir = r'/home/ljc/code/AMOS22/data/'
# path_dir = r'/nas/luojc/code/AMOS22/data'
task2_json = json.load(open(os.path.join(path_dir, 'AMOS22', 'dataset_cropped.json')))

file_path = [[os.path.join(path_dir, 'AMOS22', path_['image']),
              os.path.join(path_dir, 'AMOS22', path_['label'])]
             for path_ in task2_json['training']]

CT_train_path = file_path[0:150]
CT_valid_path = file_path[150:160]
CT_test_path = file_path[160:200]
MRI_train_path = file_path[200:230]
MRI_valid_path = file_path[230:232]
MRI_test_path = file_path[232::]
train_path = CT_train_path  # + MRI_train_path
valid_path = CT_valid_path  # + MRI_valid_path
test_path = CT_test_path  # + MRI_test_path


def cal_centre_point_5(label_arr: numpy.ndarray, label_file_path: str, r: int = 10, small_organ: list = None,
                       spacing=None):
    target = np.ones([16, 3]) * -1

    for i in np.unique(label_arr).astype(int).tolist():
        if i == 0:
            continue
        class_xyz = np.array(np.where(label_arr == i))
        x, y, z = np.mean(class_xyz, 1, dtype=int)
        target[i] = [x, y, z]
    return target


class data_set(Dataset):
    def __init__(self, file_path):
        self.paths = file_path

    def __getitem__(self, item):
        from einops import rearrange
        path_ = self.paths
        x = sitk.GetArrayFromImage(sitk.ReadImage(path_[item][0])).astype(np.int16)
        y = sitk.GetArrayFromImage(sitk.ReadImage(path_[item][1])).astype(np.int16)
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=int)
        x = resize(x, (320, 160, 160), order=1, preserve_range=True, anti_aliasing=False)
        y = resize(y, (320, 160, 160), order=0, preserve_range=True, anti_aliasing=False)
        x = rearrange(x, 'd w h -> w h d')
        y = rearrange(y, 'd w h -> w h d')
        x = self.norm(x)
        x = torch.from_numpy(x).type(torch.FloatTensor).unsqueeze_(0)
        y = torch.from_numpy(y).type(torch.FloatTensor)
        return x, y

    def __len__(self):
        return len(self.paths)

    def norm(self, x):
        #
        if np.min(x) < 0:
            # CT 图像处理
            x = np.clip(x, a_min=-175, a_max=250)
            x = (x + 175) / 425
        else:
            # MRI 图像处理
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x

    def Standardization(self, x):
        mean_x = np.mean(x)
        std_x = np.std(x)
        if std_x != 0:
            x = (x - mean_x) / std_x
        return x


def get_train_data(batch_size=1):
    data = data_set(train_path)
    return DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=True
    )


def get_valid_data():
    data = data_set(valid_path)
    return DataLoader(
        dataset=data,
        batch_size=1,
        shuffle=False
    )


def get_test_data():
    data = data_set(test_path)
    return DataLoader(
        dataset=data,
        batch_size=1,
        shuffle=False
    )


compare_path = file_path[150 - 10:150]


def get_compare_data():
    data = data_set(compare_path)
    return DataLoader(
        dataset=data,
        batch_size=1,
        shuffle=False,
    )


######################################################
##################### MODEL ##########################
class unet_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        center = self.dropout1(center)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)

        final = self.final(up1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class UnetUp3_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')  # 三线性插值

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3, 3, 1), padding_size=(1, 1, 0),
                 init_stride=(1, 1, 1)):
        # 对于z轴不做kernel
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True), )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True), )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


from torch.nn import init


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


######################################################
###########]##### LOSS_FUNCTION ######################
class BCELoss_with_weight(nn.Module):
    def __init__(self, weight):
        super(BCELoss_with_weight, self).__init__()
        self.weight = weight

    def forward(self, pred, true):
        if len(self.weight) != pred.shape[1]:
            print('shape is not mapping')
            exit()
        wei_sum = sum(self.weight)
        weight_loss = 0.
        batch_size = pred.shape[0]
        for b in range(batch_size):
            for i, class_weight in enumerate(self.weight):
                pred_i = pred[:, i]
                true_i = true[:, i]
                weight_loss += (
                        class_weight / wei_sum * F.binary_cross_entropy(pred_i, true_i, reduction='mean'))
        weight_loss = weight_loss / batch_size
        # weight_loss.requires_grad_(True)
        return weight_loss


class ComboLoss_wbce_ndice(nn.Module):
    def __init__(self, weight, alpha=0.5):
        super(ComboLoss_wbce_ndice, self).__init__()
        self.weight = weight
        self.n_classes = len(weight)
        self.CE_Crit = BCELoss_with_weight(weight)
        self.ALPHA = alpha

    def dice_loss(self, y_pred, y_true, smooth=1e-6):
        '''
        inputs:
            y_pred [batch, n_classes, x, y, z] probability
            y_true [batch, n_classes, x, y, z] one-hot code
            class_weights
            smooth = 1.0
        '''
        # smooth = 1e-6
        loss = 0.
        n_classes = y_pred.shape[1]
        batch_size = y_pred.shape[0]
        class_weights = np.asarray(self.weight, dtype=float)
        for b in range(batch_size):
            for c in range(n_classes):
                pred_flat = y_pred[b, c, ...]
                true_flat = y_true[b, c, ...]
                intersection = (pred_flat * true_flat).sum()
                # with weight
                w = class_weights[c] / class_weights.sum()
                loss += w * (1 - ((2. * intersection + smooth) /
                                  (pred_flat.sum() + true_flat.sum() + smooth)))
        return loss / batch_size

    def forward(self, pred, true):
        if len(self.weight) != pred.shape[1]:
            print('shape is not mapping')
            exit()

        DC = self.dice_loss(pred, true)
        # CE = self.CE_Crit(pred, true)
        # return (1 - self.ALPHA) * DC + self.ALPHA * CE
        return DC


class ComboLoss_wbce_dice(nn.Module):
    def __init__(self, weight, alpha=0.5):
        super(ComboLoss_wbce_dice, self).__init__()
        self.weight = weight
        self.n_classes = len(weight)
        self.CE_Crit = BCELoss_with_weight(weight)
        self.ALPHA = alpha

    def dice_loss(self, input, target, smooth=1e-6):
        # smooth = 1e-6
        # input = input.log_softmax(dim=1).exp()
        input = F.softmax(input, 1)
        loss = 0.
        batch_size = input.size(0)
        class_weights = torch.Tensor(self.weight).to(input.device)
        input = input.contiguous().view(batch_size, self.n_classes, -1)
        target = target.contiguous().view(batch_size, self.n_classes, -1)

        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth
        score = 1.0 - 2.0 * inter / union
        # score = torch.sum(score)
        score = torch.mean(score)
        return score

    def forward(self, pred, true):
        if len(self.weight) != pred.shape[1]:
            print('shape is not mapping')
            exit()
        true = one_hot(true, 16)
        true = torch.permute(true, [0, -1, 1, 2, 3])
        DC = self.dice_loss(pred, true)
        # CE = self.CE_Crit(pred, true)
        # return (1 - self.ALPHA) * DC + self.ALPHA * CE
        return DC


######################################################
#################### TRAIN_VALID #####################
def save_loss(t_loss, v_loss, v_acc, filename='tem'):
    t_loss = np.array(t_loss, dtype=np.float)
    v_loss = np.array(v_loss, dtype=np.float)
    v_acc = np.array(v_acc, dtype=np.float)
    with open(os.path.join('{}.tmp'.format(filename)), 'wb+') as f:
        pickle.dump(np.array([t_loss, v_loss, v_acc]), f)


def pic_loss_acc(filename='tem'):
    with open(os.path.join('{}.tmp'.format(filename)), 'rb+') as f:
        loss_ = pickle.load(f)
        len_train = len(loss_[0])
        train_loss, valid_loss, valid_acc = loss_[0], loss_[1], loss_[2]
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        lin1 = ax.plot([i for i in range(len_train)], train_loss, '-', label='train_loss', color='blue')
        lin2 = ax.plot([i for i in range(len_train)], valid_loss, '-', label='valid_loss', color='orange')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax2.set_ylabel('dice')
        lin3 = ax2.plot([i for i in range(len_train)], valid_acc, '-', label='dice', color='red')
        lins = lin1 + lin2 + lin3
        labs = [l.get_label() for l in lins]
        plt.legend(lins, labs, loc='best')
        # plt.show()
        plt.savefig('{}.png'.format(filename), bbox_inches='tight')
        plt.close()
        f.close()


def DICE(output, target):  # output为预测结果 target为真实结果
    smooth = 1e-5  # 防止0除

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth)


def calculate_acc(output, target, class_num, fun, is_training=False, smooth=1e-4):
    # HD_95计算库的dtype为np.numeric
    batch_size = output.size(0)
    input = output.contiguous().view(batch_size, class_num, -1)
    target = target.contiguous().view(batch_size, class_num, -1)

    inter = torch.sum(input * target, 2) + smooth
    union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

    score = torch.sum(2.0 * inter / union)
    acc = score / (float(batch_size) * float(class_num))
    return acc.item()


if __name__ == '__main__':
    print('beginning training')

    class_num = 16
    learning_rate = 1e-3
    n_epochs = 900
    batch_size = 1
    is_load = False
    device = torch.device('cpu')
    # device = torch.device('cuda:0')
    strategy = 'SSLnet'
    load_path = '/nas/luojc/code/AMOS22/src/checkpoints/new_combo_1e-3/Unet-final.pth'
    path_dir = os.path.dirname(__file__)
    # path_dir = r'/media/bj/DataFolder3/datasets/challenge_AMOS22'
    path = os.path.join(path_dir, 'checkpoints', strategy)
    if not os.path.exists(path):
        os.makedirs(path)
    # path = os.path.join(path_dir, 'checkpoints', strategy)
    model = unet_3D(n_classes=16, in_channels=1)
    loss_weight = [1, 1.02, 1.03, 1.03, 0.88, 0.87, 1.04, 0.91, 1.03, 1.01, 0.90, 0.91, 0.83, 0.85, 0.86, 0.86]
    loss_weight = [1 for _ in range(16)]
    loss1 = ComboLoss_wbce_dice(loss_weight)
    loss2 = ComboLoss_wbce_ndice(loss_weight)
    # crit = loss2

    from Dice_CE_Loss import DiceLoss, SoftCrossEntropyLoss

    loss3_dice = DiceLoss(mode='multiclass', weight=loss_weight)  ##bj
    loss4_ce = SoftCrossEntropyLoss(smooth_factor=0.0, weight=loss_weight)  ##bj

    loss5_L1 = torch.nn.SmoothL1Loss(reduction='sum')
    w_dice = 1.0
    w_ce = 1.0
    w_L1 = 0.
    # choice loss function
    # crit = loss3_dice
    crit = loss4_ce
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = get_train_data(batch_size=batch_size)
    valid_loader = get_valid_data()
    train_loss = []
    valid_loss = []
    valid_acc = []
    max_acc = 0.
    if is_load:
        model.load_state_dict(torch.load(load_path))
    writer = SummaryWriter(strategy)

    for epoch in range(1, n_epochs + 1):
        t_loss = []
        v_loss = []
        v_acc = []
        dice_loss = []
        ce_loss = []
        l1_loss = []
        model.train()
        model.to(device)
        for index, (data, GT) in enumerate(train_loader):
            break
            optimizer.zero_grad()
            # trans GT to onehot
            data = data.float().to(device)
            GT = GT.to(device)
            output = model(data)
            dc = loss3_dice(output, GT)
            ce = loss4_ce(output, GT)
            loss = w_dice * dc + w_ce * ce
            loss.backward()
            optimizer.step()
            t_loss.append(loss.item())
            dice_loss.append(dc.item())
            ce_loss.append(ce.item())
            print('{} / {}: train_loss = {}'.format(index + 1, len(train_loader), loss.item()))
        print()
        model.eval()
        # model.cpu()   ###bj  still use GPU
        with torch.no_grad():  ##bj
            table = np.zeros((16, 1))
            mask = np.ones((16, 1))
            for index, (data, GT) in enumerate(valid_loader):  ##bj
                output = model(data.float().to(device))  ##bj
                GT = GT.to(device)
                loss = w_dice * loss3_dice(output, GT)

                output = torch.argmax(output, 1).squeeze()
                GT = GT.squeeze()
                for i in range(16):
                    if i == 0:
                        continue
                    if i in torch.unique(GT):
                        SR = (output == i).type(torch.int)
                        gt = (GT == i).type(torch.int)
                        evals = float(2 * torch.sum((SR + gt) == 2)) / (float(torch.sum(SR) + torch.sum(gt)) + 1e-6)
                        mask[i] += 1
                        pass
                    else:
                        evals = 0
                        # mask[i] -= 1
                    table[i, :] += evals
                    pass
                v_loss.append(loss.item())
                print('    {} / {}: valid_loss = {}'.format(index + 1, len(valid_loader), loss.item()))
            evaluations = table / mask
            v_acc.append(np.mean(evaluations[1:-1]))
        # 每次保存最新的模型
        torch.save(model.state_dict(), os.path.join(path, 'Unet-new.pth'))
        # 保存最好的模型
        t_loss = np.mean(t_loss)
        v_loss = np.mean(v_loss)  ##bj
        v_acc = np.mean(v_acc)  ##bj
        dice_loss = np.mean(dice_loss)
        ce_loss = np.mean(ce_loss)
        print('valid_acc = {}'.format(v_acc))
        if v_acc > max_acc:
            torch.save(model.state_dict(), os.path.join(path, 'Unet-final.pth'))
            max_acc = v_acc
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(path, 'Unet-{}.pth'.format(epoch + 1)))
        writer.add_scalars('loss/train_loss', {'dice_loss': dice_loss,
                                               'ce_loss': ce_loss,
                                               'train_loss': (dice_loss + ce_loss) / 2}, epoch)
        writer.add_scalar('loss/valid_loss', v_loss, epoch)
        writer.add_scalar('info/dice', v_acc, epoch)

    writer.close()
