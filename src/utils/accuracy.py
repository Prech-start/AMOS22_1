import os.path
import copy
import torch
import numpy as np
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
# from surface_distance. import hausdorff_distance
from src.process.data_load import *
from src.model.model import *
from einops import *
import math
from absl.testing import absltest
from absl.testing import parameterized
import surface_distance
from surface_distance import metrics


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
    surface_distances = surface_distance.compute_surface_distances(true, pred, spacing_mm=(2, 2, 0.5))
    print(surface_distance.compute_average_surface_distance(surface_distances))
    return 0


def HD_95(ture, pred):
    print(0.95 * hausdorff_distance(ture, pred))
    return 0.95 * hausdorff_distance(ture, pred)


def Sensitivity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    print((intersection + smooth) / \
          (target.sum() + smooth))

    return (intersection + smooth) / \
           (target.sum() + smooth)


def calculate_acc(output, target, class_num, fun):
    # input: class_tensor
    # output shape b c w h d
    # return:
    # 将每一个通道都做一次acc计算
    acc = []
    for i in range(class_num):
        pred = copy.deepcopy(output)
        true = copy.deepcopy(target)
        # pred[pred != i] = 0
        # true[true != i] = 0
        if fun != ASD:
            pred = one_hot(torch.LongTensor(pred), 16)
            true = one_hot(torch.LongTensor(true), 16)
            pred = rearrange(pred, 'b w h d c -> b c w h d')
            true = rearrange(true, 'b w h d c -> b c w h d')
        else:
            pred = np.squeeze((pred == i), axis=0)
            true = np.squeeze((true == i).numpy(), axis=0)
        acc.append(fun(pred, true))

    # acc.append(output, target)
    return acc


def calculate_acc2(output, target, class_num, fun):
    # input: class_tensor
    # output, target's shape b c w h d
    # return: 每一个通道所计算出来的指标的值
    # 将每一个通道都做一次acc计算
    acc = []
    for i in range(class_num):
        pred = copy.deepcopy(output.data.squeeze().numpy())
        true = copy.deepcopy(target.data.squeeze().numpy())
        pred = one_hot(torch.LongTensor(pred), 16).numpy().astype(bool)
        true = one_hot(torch.LongTensor(true), 16).numpy().astype(bool)
        pred = pred[..., i]
        true = true[..., i]
        acc.append(fun(pred, true))
    # acc.append(output, target)
    return acc


from tqdm import tqdm

if __name__ == '__main__':
    test_data = data_set(False)
    data_loader = DataLoader(
        dataset=test_data,
        batch_size=1,
        pin_memory=True,
        shuffle=True
    )
    model = UnetModel(1, 16, 6)
    model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'Generalized_Dice_loss_e-3_1.pth')))
    dice_acc = []
    asd_acc = []
    hd_acc = []
    sen_acc = []
    acc = []
    with torch.no_grad():
        model.eval()
        for x, y in tqdm(data_loader):
            x = x.cpu().float()
            true = y.cpu().float().numpy()
            # j = one_hot(torch.LongTensor(true), 16)
            # i = torch.unsqueeze(i, 1)
            # true = rearrange(j, 'b w h d c -> b c w h d')
            pred = model(x)
            pred = torch.argmax(pred, dim=1)
            # asd_acc.append(calculate_acc(output=true, target=pred, class_num=16, fun=ASD))
            dice_acc.append(calculate_acc(output=pred, target=true, class_num=16, fun=DICE))
            # hd_acc.append(calculate_acc(output=true, target=pred, class_num=16, fun=HD_95))
            # sen_acc.append(calculate_acc(output=pred, target=true, class_num=16, fun=Sensitivity))
            # acc.append((np.sum(np.array(pred) == np.array(true))) / (len(pred.flatten())))
        print('done')
x = torch.Tensor(np.zeros([1, 1, 5, 5, 5]))
calculate_acc2(x, x, 16, DICE)
