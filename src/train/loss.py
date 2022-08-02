import torch
import torch.nn as nn
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def dice_loss(self, target, predictive, ep=1e-8):
        intersection = 2. * torch.sum(predictive * target) + ep
        union = torch.sum(predictive) + torch.sum(target) + ep
        loss = 1 - intersection / union
        return loss

    def forward(self, target, predictive):
        return self.dice_loss(target, predictive)


class Generalized_Dice_loss(nn.Module):
    def __init__(self, class_weight):
        super(Generalized_Dice_loss, self).__init__()
        self.class_weight = class_weight

    def Generalized_Dice_Loss(self, y_pred, y_true, class_weights, smooth=1e-6):
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
        class_weights = np.asarray(class_weights, dtype=float)
        for c in range(n_classes):  # pass 0 because 0 is background
            pred_flat = y_pred[:, c]
            true_flat = y_true[:, c]
            intersection = (pred_flat * true_flat).sum()
            # with weight
            w = class_weights[c] / class_weights.sum()
            loss += w * (1 - ((2. * intersection + smooth) /
                              (pred_flat.sum() + true_flat.sum() + smooth)))
        return loss

    # def cal_subject_level_dice(prediction, target, class_weights, class_num=2):  # class_num是你分割的目标的类别个数
    #     '''
    #     step1: calculate the dice of each category
    #     step2: remove the dice of the empty category and background, and then calculate the mean of the remaining dices.
    #     :param prediction: the automated segmentation result, a numpy array with shape of (h, w, d)
    #     :param target: the ground truth mask, a numpy array with shape of (h, w, d)
    #     :param class_num: total number of categories
    #     :return:
    #     '''
    #     eps = 1e-10
    #     empty_value = -1.0
    #     dscs = empty_value * np.ones((class_num), dtype=np.float32)
    #     class_weights = np.asarray(class_weights, dtype=float)
    #     for i in range(0, class_num):
    #         if i not in target and i not in prediction:
    #             continue
    #         target_per_class = np.where(target == i, 1, 0).astype(np.float32)
    #         prediction_per_class = np.where(prediction == i, 1, 0).astype(np.float32)
    #
    #         tp = np.sum(prediction_per_class * target_per_class)
    #         fp = np.sum(prediction_per_class) - tp
    #         fn = np.sum(target_per_class) - tp
    #         w = class_weights[i] / class_weights.sum()
    #         dsc = w * 2 * tp / (2 * tp + fp + fn + eps)
    #         dscs[i] = dsc
    #     dscs = np.where(dscs == -1.0, np.nan, dscs)
    #     subject_level_dice = np.nanmean(dscs[1:])
    #     return 1 - subject_level_dice

    def forward(self, y_pred, y_true):
        return self.Generalized_Dice_Loss(y_pred, y_true, self.class_weight)
