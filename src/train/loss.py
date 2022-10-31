import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


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

    def forward(self, y_pred, y_true):
        return self.Generalized_Dice_Loss(y_pred, y_true, self.class_weight)


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
        weight_loss.requires_grad_(True)
        return weight_loss


class ComboLoss(nn.Module):
    def __init__(self, weight):
        super(ComboLoss, self).__init__()
        self.weight = weight

    def combo(self, inputs, targets, smooth=1, eps=1e-9):
        CE_RATIO = 0.5
        ALPHA = 0.5
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        # True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        # inputs = torch.clamp(inputs, eps, 1.0 - eps)
        out = F.binary_cross_entropy(inputs, targets, reduction='mean')
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) + ((1 - CE_RATIO) * (1 - dice))

        return combo

    def forward(self, pred, true):

        if len(self.weight) != pred.shape[1]:
            print('shape is not mapping')
            exit()

        wei_sum = sum(self.weight)
        batch_size = pred.shape[0]
        loss = 0.
        for b in range(batch_size):
            for i, class_weight in enumerate(self.weight):
                pred_i = pred[:, i]
                true_i = true[:, i]
                loss += (class_weight / wei_sum) * self.combo(pred_i, true_i)
        loss = self.combo(pred, true)
        return loss


if __name__ == '__main__':
    y = torch.randn((1, 16, 256, 256, 68))
    y_ = F.softmax(y, 1)
    y_p = torch.randn((1, 16, 256, 256, 68))
    y_p_ = F.softmax(y_p, 1)
    print(y.shape)
    loss = ComboLoss(weight=[1, 2, 2, 3, 6, 6, 1, 4, 3, 4, 7, 8, 10, 5, 4, 5])
    l = loss(y_, y_p_)
