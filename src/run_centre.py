import torch
import torch.nn as nn
import os
import json
from torch.utils.data import DataLoader, Dataset
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


def cal_centre_point_4(label_arr: numpy.ndarray, label_file_path: str, r: int = 10, small_organ: list = None,
                       spacing=None):
    start_time = time.time()
    file_name = label_file_path.split('/')[-1].split('.')[0]
    path_dir = os.path.dirname(__file__)
    file_path = os.path.join(path_dir, '..', 'data', 'pointcloud6')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    save_path = os.path.join(file_path, '{}.npy'.format(file_name))
    label_arr_ = torch.nn.functional.one_hot(torch.LongTensor(label_arr), 16).permute(-1, 0, 1, 2)
    if not os.path.exists(save_path):
        centre_arr = np.zeros_like(label_arr_).astype(np.float32)
        for i in np.unique(label_arr).astype(int).tolist():
            if i == 0:
                continue
            # 对于不同大小的器官，r的取值也不一样
            # if i in small organ_list
            # r = 2/3 max_distance_from_pixel_to_centre
            class_xyz = np.array(np.where(label_arr_[i, ...] == 1))
            class_all = np.array(np.where(label_arr_[i, ...] != -1))
            class_xyz = np.r_[np.ones((1, class_xyz.shape[-1])) * i, class_xyz].astype(int)
            class_all = np.r_[np.ones((1, class_all.shape[-1])) * i, class_all].astype(int)
            c, x, y, z = np.mean(class_xyz, 1, dtype=int)
            # 求中心点坐标
            centre_arr[i, x, y, z] = 1
            # 求每个label点到中心的点的距离
            distance_for_xyz = np.linalg.norm(class_xyz.T - [i, x, y, z], axis=1)  # N * 1
            if i in small_organ:
                r = 2 / 3 * np.max(distance_for_xyz)
            distance_for_xyz = np.linalg.norm(class_all.T - [i, x, y, z], axis=1)  # N * 1
            # 筛选出distance中距离小于r的所有坐标 type:tuple
            dis = np.where(distance_for_xyz < r)
            # 筛选出到中心点距离小于r的所有坐标
            distance_list = distance_for_xyz[dis]
            # 将每个筛选出的点进行赋值
            centre_arr[tuple(class_all.T[dis].T.tolist())] = (r - distance_list) / r
            # if i in small_organ:
            #     # 第i个通道的数据取出来做spacing
            #     array = centre_arr[i, ...]
            #     IMAGE = sitk.GetImageFromArray(array)
            #     IMAGE.SetSpacing(spacing)
            #     # 保存到桌面
            #     filename = 'test.nii.gz'
            #     sitk.WriteImage(IMAGE, filename)
            #     pass
        np.save(save_path, centre_arr)
        print('data {} processing is finished, it cost {}(s)'.format(file_name, time.time() - start_time))
        return centre_arr
    # print('data {} processing is finished, it cost {}(s)'.format(file_name, time.time() - start_time))
    return np.load(save_path).astype(np.int8)


class data_set(Dataset):
    def __init__(self, file_path):
        self.paths = file_path

    def __getitem__(self, item):
        path_ = self.paths
        x = sitk.GetArrayFromImage(sitk.ReadImage(path_[item][0])).astype(np.int16)
        y = sitk.GetArrayFromImage(sitk.ReadImage(path_[item][1])).astype(np.int16)
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=int)
        x = resize(x, (320, 160, 160), order=1, preserve_range=True, anti_aliasing=False)
        y = resize(y, (320, 160, 160), order=0, preserve_range=True, anti_aliasing=False)
        x = self.norm(x)
        # x = self.Standardization(x)
        # z = resize(z, (64, 256, 256), order=0, preserve_range=True, anti_aliasing=False)
        small_organ = [4, 5, 10, 11, 12, 13, 14, 15]
        z = cal_centre_point_4(y.squeeze(), path_[item][1], r=10, small_organ=small_organ)
        x = torch.from_numpy(x).type(torch.FloatTensor).unsqueeze_(0)
        y = torch.from_numpy(y).type(torch.FloatTensor)
        z = torch.from_numpy(z).type(torch.FloatTensor)
        return x, y, z

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
        shuffle=True
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
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                                stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        x = self.batch_norm(self.conv3d(x))
        # x = self.conv3d(x)
        x = F.elu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, model_depth=4, pool_size=2):
        super(EncoderBlock, self).__init__()
        self.root_feat_maps = 4
        self.num_conv_blocks = 2
        # self.module_list = nn.ModuleList()
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.root_feat_maps
            for i in range(self.num_conv_blocks):
                # print("depth {}, conv {}".format(depth, i))
                if depth == 0:
                    # print(in_channels, feat_map_channels)
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
                else:
                    # print(in_channels, feat_map_channels)
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
            if depth == model_depth - 1:
                break
            else:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling

    def forward(self, x):
        down_sampling_features = []
        for k, op in self.module_dict.items():
            if k.startswith("conv"):
                x = op(x)
                # print(k, x.shape)
                if k.endswith("1"):
                    down_sampling_features.append(x)
            elif k.startswith("max_pooling"):
                x = op(x)
                # print(k, x.shape)

        return x, down_sampling_features


class DecoderBlock(nn.Module):
    def __init__(self, out_channels, model_depth=4):
        super(DecoderBlock, self).__init__()
        self.num_conv_blocks = 2
        self.num_feat_maps = 4
        # user nn.ModuleDict() to store ops
        self.module_dict = nn.ModuleDict()

        for depth in range(model_depth - 2, -1, -1):
            # print(depth)
            feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps
            # print(feat_map_channels * 4)
            self.deconv = ConvTranspose(in_channels=feat_map_channels * 4, out_channels=feat_map_channels * 4)
            self.module_dict["deconv_{}".format(depth)] = self.deconv
            for i in range(self.num_conv_blocks):
                if i == 0:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 6, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
                else:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
            if depth == 0:
                self.final_conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=out_channels)
                self.module_dict["final_conv"] = self.final_conv

    def forward(self, x, down_sampling_features):
        """
        :param x: inputs
        :param down_sampling_features: feature maps from encoder path
        :return: output
        """
        for k, op in self.module_dict.items():
            if k.startswith("deconv"):
                x = op(x)
                x = torch.cat((down_sampling_features[int(k[-1])], x), dim=1)
            elif k.startswith("conv"):
                x = op(x)
            else:
                x = op(x)
        return x


class UnetModel(nn.Module):

    def __init__(self, in_channels, out_channels, model_depth=4, final_activation="softmax"):
        super(UnetModel, self).__init__()
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth)
        self.decoder = DecoderBlock(out_channels=out_channels, model_depth=model_depth)
        # self.decoder_centre = DecoderBlock(out_channels=1, model_depth=model_depth)
        self.centre_head = ConvBlock(out_channels, out_channels)
        if final_activation == "sigmoid":
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = nn.Softmax(dim=1)

    def forward(self, x):
        x, downsampling_features = self.encoder(x)
        seg_x = self.decoder(x, downsampling_features)
        # seg_x = self.sigmoid(seg_x)   ##bj
        # # print("Final output shape: ", x.shape)
        centre = self.centre_head(seg_x)
        centre = self.sigmoid(centre)
        return seg_x, centre


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=2, padding=1, output_padding=1):
        super(ConvTranspose, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=k_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   output_padding=output_padding)

    def forward(self, x):
        return self.conv3d_transpose(x)


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
    n_epochs = 300
    batch_size = 1
    is_load = False
    device = torch.device('cpu')
    # device = torch.device('cuda:0')
    strategy = 'centre_final_01'
    load_path = '/nas/luojc/code/AMOS22/src/checkpoints/new_combo_1e-3/Unet-final.pth'
    path_dir = os.path.dirname(__file__)
    # path_dir = r'/media/bj/DataFolder3/datasets/challenge_AMOS22'
    path = os.path.join(path_dir, 'checkpoints', strategy)
    if not os.path.exists(path):
        os.makedirs(path)
    # path = os.path.join(path_dir, 'checkpoints', strategy)
    model = UnetModel(1, 16, 6)
    loss_weight = [1, 1.02, 1.03, 1.03, 0.88, 0.87, 1.04, 0.91, 1.03, 1.01, 0.90, 0.91, 0.83, 0.85, 0.86, 0.86]
    loss1 = ComboLoss_wbce_dice(loss_weight)
    loss2 = ComboLoss_wbce_ndice(loss_weight)
    # crit = loss2

    from Dice_CE_Loss import DiceLoss, SoftCrossEntropyLoss

    loss3_dice = DiceLoss(mode='multiclass', weight=loss_weight)  ##bj
    loss4_ce = SoftCrossEntropyLoss(smooth_factor=0.0, weight=loss_weight)  ##bj

    loss5_L1 = torch.nn.SmoothL1Loss()
    w_dice = 1.0
    w_ce = 1.0
    w_L1 = 0.1
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

    wind_loss = Visdom(env=strategy)
    wind_dice = Visdom(env=strategy)
    wind_l1 = Visdom(env=strategy)

    wind_dice.line([[0.]],  # Y的第一个点的坐标
                   [0.],  # X的第一个点的坐标
                   win='dice',  # 窗口的名称
                   opts=dict(title='dice', legend=['dice'])  # 图像的标例
                   )

    for epoch in range(1, n_epochs + 1):
        w_L1 = ((epoch - n_epochs) ** 2) / ((1 - n_epochs) ** 2)
        t_loss = []
        v_loss = []
        v_acc = []
        dice_loss = []
        ce_loss = []
        l1_loss = []
        model.train()
        model.to(device)
        if epoch == 200:
            learning_rate = 1e-4
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for index, (data, GT, C_GT) in enumerate(train_loader):
            save_path = os.path.join('output', 'AMOS_CENTRE', str(index))
            for i in range(data.squeeze().shape[0]):
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.imsave(os.path.join(save_path, str(i) + '.png'), data.squeeze()[i, ...], cmap='gray')
                GT[GT != 4] = 0
                plt.imsave(os.path.join(save_path, str(i) + '_GT.png'), GT.squeeze()[i, ...], cmap='gray')
                for j in range(C_GT.squeeze().shape[0]):
                    save_path_ = os.path.join(save_path, str(j))
                    if not os.path.exists(save_path_):
                        os.makedirs(save_path_)
                    plt.imsave(os.path.join(save_path_, str(i) + '_CGT.png'), C_GT.squeeze()[j, i, ...], cmap='gray')
            print('finish')
            pass
            continue
            # train_data
            optimizer.zero_grad()
            # trans GT to onehot
            data = data.float().to(device)
            GT = GT.to(device)
            C_GT = C_GT.to(device)
            # GT = one_hot(GT, 16)
            # GT = torch.permute(GT, ( 0, 4, 1, 2, 3))
            # # GT = rearrange(GT, 'b d w h c -> b c d w h')
            # training param
            output, C_output = model(data)
            # loss_ = loss1(output, GT)
            # print(loss_)
            # dcL = loss3_dice(output, GT)
            dc = loss3_dice(output, GT)
            ce = loss4_ce(output, GT)
            l1 = loss5_L1(C_output, C_GT)
            loss = w_dice * dc + w_ce * ce + w_L1 * l1
            # loss = crit(output, GT.float())
            loss.backward()
            optimizer.step()
            t_loss.append(loss.item())
            dice_loss.append(dc.item())
            ce_loss.append(ce.item())
            l1_loss.append(l1.item())
            # print('\r \t {} / {}:train_loss = {}'.format(index + 1, len(train_loader), loss.item()), end="")
            print('{} / {}: train_loss = {}'.format(index + 1, len(train_loader), loss.item()))
        print()
        model.eval()
        # model.cpu()   ###bj  still use GPU
        with torch.no_grad():  ##bj
            for index, (data, GT, _) in enumerate(valid_loader):  ##bj
                # valid data
                # GT = one_hot(GT, 16)
                # target = torch.permute(GT, ( 0, 4, 1, 2, 3))
                # # target = rearrange(GT, 'b d w h c -> b c d w h')
                # training param
                output, _ = model(data.float().to(device))  ##bj
                GT = GT.to(device)
                loss = w_dice * loss3_dice(output, GT) + w_ce * loss4_ce(output, GT)

                output = output.log_softmax(dim=1).exp()  ##bj
                GT = one_hot(GT.to(torch.long), 16)
                target = torch.permute(GT, (0, 4, 1, 2, 3))
                v_acc.append(
                    calculate_acc(output, target, class_num=16, fun=DICE,
                                  is_training=True))
                v_loss.append(loss.item())
                print('    {} / {}: valid_loss = {}'.format(index + 1, len(valid_loader), loss.item()))
        # 每次保存最新的模型
        torch.save(model.state_dict(), os.path.join(path, 'Unet-new.pth'))
        # 保存最好的模型
        t_loss = np.mean(t_loss)
        v_loss = np.mean(v_loss)  ##bj
        v_acc = np.mean(v_acc)  ##bj
        dice_loss = np.mean(dice_loss)
        ce_loss = np.mean(ce_loss)
        l1_loss = np.mean(l1_loss)
        print('valid_acc = {}'.format(v_acc))
        if v_acc > max_acc:
            torch.save(model.state_dict(), os.path.join(path, 'Unet-final.pth'))
            max_acc = v_acc
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(path, 'Unet-{}.pth'.format(epoch + 1)))
        # train_loss.append(t_loss)
        # valid_loss.append(v_loss)
        # valid_acc.append(v_acc)
        # # 保存训练的loss
        wind_loss.line([[l1_loss]],  # Y的第一个点的坐标
                       [epoch],  # X的第一个点的坐标
                       win='l1_loss',  # 窗口的名称
                       update='append',
                       opts=dict(title='l1_loss', legend=['l1_loss'])
                       )
        wind_l1.line([[dice_loss, ce_loss, (dice_loss + ce_loss) / 2, v_loss]],  # Y的第一个点的坐标
                       [epoch],  # X的第一个点的坐标
                       win='train&valid_loss',  # 窗口的名称
                       update='append',
                       opts=dict(title='train_loss', legend=['dice_loss', 'ce_loss', 'train_loss', 'valid_loss'])
                       )

        wind_dice.line([[v_acc]],  # Y的第一个点的坐标
                       [epoch],  # X的第一个点的坐标
                       win='dice',  # 窗口的名称
                       update='append')  # 图像的标例
        time.sleep(0.5)
