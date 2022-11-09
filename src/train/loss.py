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
        # weight_loss.requires_grad_(True)
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


class ComboLoss2(nn.Module):
    def __init__(self, weight):
        super(ComboLoss2, self).__init__()
        self.weight = weight
        self.n_classes = len(weight)
        self.CE_Crit = nn.CrossEntropyLoss(weight=torch.Tensor(self.weight))

    def Generalized_Dice_Loss(self, input, target, smooth=1e-6):
        '''
        inputs:
            y_pred [batch, n_classes, x, y, z] probability
            y_true [batch, n_classes, x, y, z] one-hot code
            class_weights
            smooth = 1.0
        '''
        # smooth = 1e-6
        loss = 0.
        batch_size = input.size(0)
        class_weights = torch.Tensor(self.weight).to(input.device)
        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        target = target.contiguous().view(batch_size, self.n_classes, -1)

        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union * class_weights / class_weights.sum())
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))
        return score

    def forward(self, pred, true, ALPHA=0.5):
        if len(self.weight) != pred.shape[1]:
            print('shape is not mapping')
            exit()

        DC = self.Generalized_Dice_Loss(pred, true, self.weight)
        CE = self.CE_Crit(pred, torch.argmax(true, 1))
        return (1 - ALPHA) * DC + ALPHA * CE


class ComboLoss3(nn.Module):
    def __init__(self, weight, alpha=0.5):
        super(ComboLoss3, self).__init__()
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
        CE = self.CE_Crit(pred, true)
        return (1 - self.ALPHA) * DC + self.ALPHA * CE


class One_Hot(nn.Module):
    def __init__(self, depth=2):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth)

    def forward(self, X_in):
        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth])
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0, X_in)).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes

    def forward(self, input, target):
        smooth = 0.1
        batch_size = input.size(0)

        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        # print("max::::", torch.max(input), "min::::", torch.min(input))
        # print("max::::", torch.max(target), "min::::", torch.min(target))
        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score


if __name__ == '__main__':
    y = torch.randn((1, 16, 256, 256, 68))
    y_ = F.softmax(y, 1)
    y_p = torch.randn((1, 16, 256, 256, 68))
    y_p_ = F.softmax(y_p, 1)
    print(y.shape)
    loss = ComboLoss(weight=[1, 2, 2, 3, 6, 6, 1, 4, 3, 4, 7, 8, 10, 5, 4, 5])
    l = loss(y_, y_p_)
    print(l.item())
