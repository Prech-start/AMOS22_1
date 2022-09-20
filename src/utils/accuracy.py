import copy

import torch
from torch.nn.functional import one_hot
# from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
from src.process.data_load import *
from src.model.model import *
import surface_distance
from medpy.metric import binary
import numpy as np


def DICE(output, target):  # output为预测结果 target为真实结果
    smooth = 1e-5  # 防止0除

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth)


def AJI(true, pred):
    """AJI+, an AJI version with maximal unique pairing to obtain overall intersecion.
    Every prediction instance is paired with at most 1 GT instance (1 to 1) mapping, unlike AJI
    where a prediction instance can be paired against many GT instances (1 to many).
    Remaining unpaired GT and Prediction instances will be added to the overall union.
    The 1 to 1 mapping prevents AJI's over-penalisation from happening.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.

    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    #### Munkres pairing to find maximal unique pairing
    paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
    ### extract the paired cost and remove invalid pair
    paired_iou = pairwise_iou[paired_true, paired_pred]
    # now select all those paired with iou != 0.0 i.e have intersection
    paired_true = paired_true[paired_iou > 0.0]
    paired_pred = paired_pred[paired_iou > 0.0]
    paired_inter = pairwise_inter[paired_true, paired_pred]
    paired_union = pairwise_union[paired_true, paired_pred]
    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    overall_inter = paired_inter.sum()
    overall_union = paired_union.sum()
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()
    #
    aji_score = overall_inter / overall_union
    return aji_score


def ASD(true, pred):
    surface_distances = surface_distance.compute_surface_distances(true, pred, spacing_mm=(1, 1, 1))
    return surface_distance.compute_average_surface_distance(surface_distances)


def HD_95(ture, pred):
    return binary.hd95(ture, pred)


def Sensitivity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
           (target.sum() + smooth)


def calculate_acc(output, target, class_num, fun, is_training=False):
    # input: class_tensor
    # output, target's shape b c w h d
    # return: 每一个通道所计算出来的指标的值
    # 将每一个通道都做一次acc计算
    acc = []
    if len(torch.unique(output)) != len(torch.unique(target)):
        print('error')
    # HD_95计算库的dtype为np.numeric
    dtype_ = bool
    for i in range(class_num):
        # 跳过background
        if i == 0:
            continue
        if is_training:
            break
        pred = copy.deepcopy(output.data.squeeze().numpy())
        true = copy.deepcopy(target.data.squeeze().numpy())
        pred = one_hot(torch.LongTensor(pred), 16).numpy().astype(dtype_)
        true = one_hot(torch.LongTensor(true), 16).numpy().astype(dtype_)
        pred = pred[..., i]
        true = true[..., i]
        acc.append(fun(pred, true))
    pred = output.data.squeeze().numpy().astype(bool)
    true = target.data.squeeze().numpy().astype(bool)
    acc.append(fun(pred, true))
    return acc


def calculate_dice_all(test_loader, model):
    from einops import rearrange
    model.cpu()
    dice = []
    class_num = 16
    dtype_ = bool

    with torch.no_grad():
        for x, y in tqdm(test_loader):
            acc = []
            target = torch.LongTensor(y.long())
            output = model(x.float())
            output = torch.argmax(output, 1)
            pred = copy.deepcopy(output.data.squeeze().numpy())
            true = copy.deepcopy(target.data.squeeze().numpy())
            pred = one_hot(torch.LongTensor(pred), 16).numpy().astype(dtype_)
            true = one_hot(torch.LongTensor(true), 16).numpy().astype(dtype_)
            uni_list = torch.unique(target)
            all_list = [i for i in range(16)]
            for i in range(class_num):
                # 跳过background
                if i == 0:
                    continue
                if i not in uni_list and i in all_list:
                    temp = -1
                else:
                    temp = DICE(pred[..., i], true[..., i])
                acc.append(temp)
            pred = output.data.squeeze().numpy().astype(bool)
            true = target.data.squeeze().numpy().astype(bool)
            acc.append(DICE(pred, true))
            dice.append(acc)
    return np.array(dice)


def cal_nnunet_dice():
    from glob import glob
    dice = []
    class_num = 16
    dtype_ = bool
    infer_path = "/home/ljc/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task051_AMOS/inferTs/*"  # 推理结果地址
    label_path = "/home/ljc/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task051_AMOS/imagesTl/*"  # 测试集label地址
    infer = sorted(glob(infer_path))[:-1]
    label = sorted(glob(label_path))
    score_avg = 0
    for i in range(len(label)):
        acc = []
        inf, lab = infer[i], label[i]
        inf, lab = sitk.ReadImage(inf, sitk.sitkFloat32), sitk.ReadImage(lab, sitk.sitkFloat32)
        inf, lab = sitk.GetArrayFromImage(inf).astype(np.long), sitk.GetArrayFromImage(lab).astype(np.long)
        inf, lab = torch.from_numpy(inf), torch.from_numpy(lab)
        output, target = inf.unsqueeze(0).unsqueeze(0), lab.unsqueeze(0).unsqueeze(0)
        pred = one_hot(torch.LongTensor(output), 16).numpy().astype(dtype_)
        true = one_hot(torch.LongTensor(target), 16).numpy().astype(dtype_)
        uni_list = torch.unique(target)
        all_list = [i for i in range(16)]
        for i in range(class_num):
            # 跳过background
            if i == 0:
                continue
            if i not in uni_list and i in all_list:
                temp = -1
            else:
                temp = DICE(pred[..., i], true[..., i])
            acc.append(temp)
        pred = output.data.squeeze().numpy().astype(bool)
        true = target.data.squeeze().numpy().astype(bool)
        acc.append(DICE(pred, true))
        dice.append(acc)
    return np.array(dice)




from tqdm import tqdm
import pandas as pd
from src.process.task2_data_loader import get_dataloader
#
# model = UnetModel(1, 16, 6)
# model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save_task2', 'Unet-180.pth')))
#
# acc = calculate_dice_all(get_dataloader(False), model)
# dices = []
# for n_class in range(0, 16):
#     c_dices = acc[..., n_class]
#     c_dice = np.mean(c_dices[np.where(c_dices != -1)])
#     dices.append(c_dice)
# dices = np.array(dices)
# print(dices)

# if __name__ == '__main__':
#     data_loader = get_dataloader(is_train=False, batch_size=1)
#     dict_ = {
#         # "0": "background",
#         "1": "spleen",
#         "2": "right kidney",
#         "3": "left kidney",
#         "4": "gall bladder",
#         "5": "esophagus",
#         "6": "liver",
#         "7": "stomach",
#         "8": "aorta",
#         "9": "postcava",
#         "10": "pancreas",
#         "11": "right adrenal gland",
#         "12": "left adrenal gland",
#         "13": "duodenum",
#         "14": "bladder",
#         "15": "prostate/uterus",
#         "16": "total"
#     }
#     model = UnetModel(1, 16, 6)
#     model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save_task2', 'Unet-240.pth')))
#     dice_acc = []
#     asd_acc = []
#     hd_acc = []
#     sen_acc = []
#     acc = []
#     error_num = 0
#     with torch.no_grad():
#         model.eval()
#     for x, y in data_loader:
#         x = x.cpu().float()
#         true = y.cpu().float()
#         pred = model(x)
#         pred = torch.argmax(pred, dim=1)
#         print(torch.unique(pred))
#         if np.unique(pred).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15]:
#             error_num += 1
#             continue
#         if len(np.unique(true)) != 16:
#             error_num += 1
#             continue
#         if len(np.unique(pred)) != 16:
#             error_num += 1
#             continue
#         # # ok
#         asd_acc.append(calculate_acc(output=true, target=pred, class_num=16, fun=ASD))
#         # # ok
#         dice_acc.append(calculate_acc(output=pred, target=true, class_num=16, fun=DICE))
#         # ok
#         hd_acc.append(calculate_acc(output=true, target=pred, class_num=16, fun=HD_95))
#         # ok
#         sen_acc.append(calculate_acc(output=pred, target=true, class_num=16, fun=Sensitivity))
#     # 将每一个指标存进execl
#     asd_acc = np.array(asd_acc)
#     dice_acc = np.array(dice_acc)
#     hd_acc = np.array(hd_acc)
#     sen_acc = np.array(sen_acc)
#     acc_matrix = [dice_acc, asd_acc[:, :, 0], asd_acc[:, :, 1], hd_acc, sen_acc]
#     # acc_matrix.shape = acc_num, item_num, class_num
#     acc_matrix = np.array(acc_matrix).mean(axis=1).T
#     data_matrix = pd.DataFrame(acc_matrix)
#     data_matrix.columns = ['DICE', 'ASD_GT2PRED', 'ASD_PRED2GT', 'HD_95', 'SENSITIVITY']
#     data_matrix.index = dict_.values()
#     writer = pd.ExcelWriter('accuracy_weight_task2_240.xlsx')
#     data_matrix.to_excel(writer, 'page_1', float_format='%.5f')
#     writer.save()
#     print('error number:{}/{}'.format(error_num, len(data_loader)))
#     print('done')
#     # x = torch.Tensor(np.zeros([1, 1, 5, 5, 5]))
#     # calculate_acc2(x, x, 16, DICE)
if __name__ == '__main__':
    data_loader = get_dataloader(is_train=False, batch_size=1)
    dict_ = {
        # "0": "background",
        "1": "spleen",
        "2": "right kidney",
        "3": "left kidney",
        "4": "gall bladder",
        "5": "esophagus",
        "6": "liver",
        "7": "stomach",
        "8": "aorta",
        "9": "postcava",
        "10": "pancreas",
        "11": "right adrenal gland",
        "12": "left adrenal gland",
        "13": "duodenum",
        "14": "bladder",
        "15": "prostate/uterus",
        "16": "total"
    }
    model = UnetModel(1, 16, 6)
    model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save_task2', 'Unet-220.pth')))

    acc_nnunet = cal_nnunet_dice()
    acc = calculate_dice_all(get_dataloader(False), model)
    dices = []
    dices_nnunet = []
    for n_class in range(0, 16):
        c_dices = acc[..., n_class]
        c_dices_nnunet = acc_nnunet[..., n_class]
        c_dice = np.mean(c_dices[np.where(c_dices != -1)])
        c_dice_nnunet = np.mean(c_dices_nnunet[np.where(c_dices_nnunet != -1)])
        dices.append(c_dice)
        dices_nnunet.append(c_dice_nnunet)
    dices = np.array(dices)
    dices_nnunet = np.array(dices_nnunet)
    # 将每一个指标存进execl
    acc_matrix = [dices, dices_nnunet]
    # acc_matrix.shape = acc_num, item_num, class_num
    acc_matrix = np.array(acc_matrix).T
    data_matrix = pd.DataFrame(acc_matrix)
    data_matrix.columns = ['DICE', 'DICE_nnunet']
    data_matrix.index = dict_.values()
    writer = pd.ExcelWriter('accuracy_weight_task2_220_dice.xlsx')
    data_matrix.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()
    print('done')
