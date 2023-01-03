import numpy as np
import torch
from tqdm import tqdm

# from Dice_CE_Loss import DiceLoss
from torch.nn.modules.loss import _Loss
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss
import torch.nn as nn

ACCURACY = "accuracy"
DICE = "dice"
F1 = "f1"
SENSITIVITY = "SENSITIVITY"
SPECIFICITY = "SPECIFICITY"
PRECISION = "PRECISION"
JS = "js"
EVALUATIONS = [ACCURACY, DICE, F1, SENSITIVITY, SPECIFICITY, PRECISION, JS]


def GetEvaluation(SR: Tensor, GT: Tensor, EVALS: list = EVALUATIONS):
    SR = SR.type(torch.int)
    GT = GT.type(torch.int)
    TP = ((SR == 1) * 1 + (GT == 1) * 1) == 2
    FN = ((SR == 0) * 1 + (GT == 1) * 1) == 2
    TN = ((SR == 0) * 1 + (GT == 0) * 1) == 2
    FP = ((SR == 1) * 1 + (GT == 0) * 1) == 2
    acc = 0.
    dice = 0.
    f1 = 0.
    sensitivity = 0.
    specificity = 0.
    precision = 0.
    js = 0.
    for eval in EVALS:
        assert eval in EVALUATIONS
        if eval == ACCURACY:
            acc = float(torch.sum(TP + TN)) / (float(torch.sum(TP + FN + TN + FP)) + 1e-6)
        if eval == SENSITIVITY:
            sensitivity = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
        if eval == SPECIFICITY:
            specificity = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)
        if eval == PRECISION:
            precision = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)
        if eval == F1:
            SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
            PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)
            f1 = 2 * SE * PC / (SE + PC + 1e-6)
        if eval == JS:
            Inter = torch.sum((SR + GT) == 2)
            Union = torch.sum((SR + GT) >= 1)
            js = float(Inter) / (float(Union) + 1e-6)
        if eval == DICE:
            Inter = torch.sum((SR + GT) == 2)
            dice = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)
    return [acc, sensitivity, specificity, precision, f1, js, dice]


from run_centre import get_test_data, UnetModel, get_train_data, get_compare_data


def cal_dice_loss(model_path: str, dataloader):
    model = UnetModel(1, 16, 6)
    # model.cpu()
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    # dc = DiceLoss(mode='multiclass')
    table = np.zeros((16, len(EVALUATIONS)))
    mask = np.ones((16, 1))
    for ori, gt, _ in tqdm(dataloader):
        pred,_ = model(ori.float())
        pred = torch.argmax(pred, 1).squeeze()
        for i in range(16):
            if i == 0:
                continue
            if i in torch.unique(gt):
                evals = GetEvaluation(pred == i, gt == i)
                mask[i] += 1
                pass
            else:
                evals = [0 for _ in range(len(EVALUATIONS))]
                # mask[i] -= 1
            table[i, :] += evals
            pass
        pass
    evaluations = table / mask
    # print(evaluations)
    return evaluations


if __name__ == '__main__':
    print('begin')
    model_path = '/home/ljc/code/AMOS22/src/checkpoints/centre_final_02/Unet-final.pth'
    cal_dice_loss(model_path=model_path, dataloader=get_test_data())
    compare_evals = cal_dice_loss(model_path=model_path, dataloader=get_compare_data())
    valid_evals = cal_dice_loss(model_path=model_path, dataloader=get_test_data())
    compare_array = np.concatenate([compare_evals, valid_evals], axis=1)
    print('finish')
