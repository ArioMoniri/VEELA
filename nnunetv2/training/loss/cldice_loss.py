import torch
import torch.nn as nn
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss

import torch
import torch.nn.functional as F
import torch.nn as nn
from monai.losses import DiceLoss
import warnings
import numpy as np
import sys

class CompoundLoss(nn.Module):
    def __init__(self, loss1, loss2=None, alpha1=1., alpha2=0.):
        super(CompoundLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, y_pred, y_true, y_true_sk=None):
        l1 = self.loss1(y_pred, y_true, y_true_sk)
        if self.alpha2 == 0 or self.loss2 is None:
            return self.alpha1*l1
        l2 = self.loss2(y_pred, y_true, y_true_sk)
        return self.alpha1*l1 + self.alpha2 * l2

# just a wrapper to make sure you are not using softmax before the loss computation
class DSCLoss(nn.Module):
    def __init__(self, include_background=False):
        super(DSCLoss, self).__init__()
        self.loss = DiceLoss(softmax=True, include_background=include_background)
        self.check_softmax = True

    def forward(self, y_pred, y_true, y_true_sk=None):
        if self.check_softmax:
            if y_pred.softmax(dim=1).flatten(2).sum(dim=1).mean(dim=1).mean() != 1.0:
                # flatten(2) flattens all after dim=2, sum over classes, take mean to get per-batch values, & take mean
                warnings.warn('check you did not apply softmax before loss computation')
            else: self.check_softmax = False
        return self.loss(y_pred, y_true)

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
    def forward(self, y_pred, y_true, y_true_sk=None):
        return self.loss(y_pred, y_true)


def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1),(1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img,(3, 3, 3),(1, 1, 1), (1, 1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iter_):
    img1 = soft_open(img)
    skel = F.relu(img-img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img-img1)
        skel = skel + F.relu(delta-skel*delta)
    return skel

class clDiceLoss(torch.nn.Module):
    def __init__(self, iters=3, smooth=1., include_background=False):
        super(clDiceLoss, self).__init__()
        self.iters = iters
        self.smooth = smooth
        self.include_background = include_background
        self.check_softmax = True

    def forward(self, y_pred, y_true, y_true_sk=True):
        if self.check_softmax:
            if y_pred.softmax(dim=1).flatten(2).sum(dim=1).mean(dim=1).mean() != 1.0:
                warnings.warn('check you did not apply softmax before loss computation')
            else: self.check_softmax = False
        y_pred = y_pred.softmax(dim=1)
        y_pred_sk = soft_skel(y_pred, self.iters)
        if y_true_sk is None:
            y_true_sk = soft_skel(y_true, self.iters)

        tprec = (torch.sum(torch.multiply(y_pred_sk, y_true)[:, 1:, ...]) + self.smooth)/(torch.sum(y_pred_sk[:, 1:, ...]) + self.smooth)
        tsens = (torch.sum(torch.multiply(y_true_sk, y_pred)[:, 1:, ...]) + self.smooth)/(torch.sum(y_true_sk[:, 1:, ...]) + self.smooth)
        cl_dice = 1. - 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice

class clCELoss(torch.nn.Module):
    def __init__(self, iters=3, smooth=1., include_background=False):
        super(clCELoss, self).__init__()
        self.iters = iters
        self.smooth = smooth
        self.include_background = include_background
        self.check_softmax = True

    def forward(self, y_pred, y_true, y_true_sk=None):

        if self.check_softmax:
            if y_pred.softmax(dim=1).flatten(2).sum(dim=1).mean(dim=1).mean() != 1.0:
                warnings.warn('check you did not apply softmax before loss computation')
            else: self.check_softmax = False
        l_unred = torch.nn.functional.cross_entropy(y_pred, y_true, reduction='none')
        y_pred = y_pred.softmax(dim=1)
        y_pred_sk = soft_skel(y_pred, self.iters)
        if y_true_sk is None:
            y_true_sk = soft_skel(y_true, self.iters)

        tprec = torch.mul(l_unred, y_true_sk[:, 1]).mean()
        tsens = torch.mul(l_unred, y_pred_sk[:, 1]).mean()
        cl_ce = (tprec+tsens)
        return cl_ce

def get_loss(loss1, loss2=None, alpha1=1., alpha2=0.):
    if loss1 == loss2 and alpha2 != 0.:
        warnings.warn('using same loss twice, you sure?')
    loss_dict = dict()
    loss_dict['ce'] = CELoss()
    loss_dict['dice'] = DSCLoss()
    loss_dict['cedice'] = CompoundLoss(CELoss(), DSCLoss(), alpha1=1., alpha2=1.)
    loss_dict['cldice'] = clDiceLoss()
    loss_dict['clce'] = clCELoss()
    loss_dict[None] = None

    loss_fn = CompoundLoss(loss_dict[loss1], loss_dict[loss2], alpha1, alpha2)

    return loss_fn





def soft_dice(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    intersection = torch.sum((y_true * y_pred)[:, 1:, ...])
    coeff = (2.0 * intersection + smooth) / (torch.sum(y_true[:, 1:, ...]) + torch.sum(y_pred[:, 1:, ...]) + smooth)
    soft_dice: torch.Tensor = 1.0 - coeff
    return soft_dice



class dice_cldice_loss(nn.Module):
    def __init__(self, iter_=3, smooth=1.0, weight_dice=1, weight_cldice=1):
        super(dice_cldice_loss, self).__init__()
        self.iter_ = iter_
        self.smooth = smooth
        self.weight_dice=weight_dice
        self.weight_cldice=weight_cldice

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:#def forward(y_true, y_pred):
        y_pred=y_pred.softmax(dim=1)

        y_true_oh = torch.zeros(y_pred.shape, device=y_pred.device)
        y_true_oh.scatter_(1, y_true.long(), 1) # nnUNet-training-loss-dice.py

        dice = soft_dice(y_true_oh, y_pred, self.smooth)

        skel_pred = soft_skel(y_pred, self.iter_)
        skel_true = soft_skel(y_true_oh, self.iter_)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true_oh)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)    
        cl_dice = 1.0- 2.0*(tprec*tsens)/(tprec+tsens)
        #total_loss: torch.Tensor = (1.0 - self.alpha) * dice + self.alpha * cl_dice
        result = self.weight_dice * dice + self.weight_cldice * cl_dice
        return result


class dice_clCE_loss(nn.Module):
    def __init__(self, iter_=3, smooth=1.0, weight_dice=1, weight_clCE=1):
        super(dice_clCE_loss, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.weight_clCE = weight_clCE
        self.weight_dice = weight_dice

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor: # def forward(y_true, y_pred):
        y_true_oh = torch.zeros(y_pred.shape, device=y_pred.device)
        y_true_oh.scatter_(1, y_true.long(), 1) # nnUNet-training-loss-dice.py

        cross_ent = torch.nn.functional.cross_entropy(y_pred, y_true_oh, reduction="none")
        y_pred = y_pred.softmax(dim=1)

        dice = soft_dice(y_true_oh, y_pred, self.smooth)

        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true_oh, self.iter)
        tprec = torch.mul(cross_ent, skel_true[:,1]).mean()
        tsens = torch.mul(cross_ent, skel_pred[:,1]).mean()
        cl_ce = (tprec+tsens)
        result = self.weight_dice * dice + self.weight_clCE * cl_ce
        return result



#CE + algo
class CE_cldice_loss(nn.Module):
    def __init__(self, ce_kwargs, iter_=3, smooth=1.0, weight_ce=1, weight_cldice=1, ignore_label=None):
        super(CE_cldice_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.iter_ = iter_
        self.smooth = smooth
        self.weight_cldice = weight_cldice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None) else 0

        target_oh = torch.zeros(net_output.shape, device=net_output.device)
        target_oh.scatter_(1, target.long(), 1) # nnUNet-training-loss-dice.py

        net_output=net_output.softmax(dim=1)
        skel_true = soft_skel(target_oh, self.iter_)
        skel_pred = soft_skel(net_output, self.iter_)
        #tprec = (torch.sum(torch.multiply(skel_pred, target[:, 0])[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)
        tprec = (torch.sum(torch.multiply(skel_pred, target_oh)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, net_output)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)
        cl_dice = 1.0- 2.0*(tprec*tsens)/(tprec+tsens)

        ##Total loss computation##
        result = self.weight_ce * ce_loss + self.weight_cldice * cl_dice
        return result

class CE_clCE_loss(nn.Module):
    def __init__(self, ce_kwargs, iter_=3, weight_ce=1, weight_clCE=1):
        super(CE_clCE_loss, self).__init__()
        self.iter = iter_
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.weight_clCE = weight_clCE
        self.weight_ce = weight_ce

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor: # def forward(y_true, y_pred):
        y_true_oh = torch.zeros(y_pred.shape, device=y_pred.device)
        y_true_oh.scatter_(1, y_true.long(), 1) # nnUNet-training-loss-dice.py

        ce_loss = self.ce(y_pred, y_true[:, 0])
        cross_ent = torch.nn.functional.cross_entropy(y_pred, y_true_oh, reduction="none")
        y_pred = y_pred.softmax(dim=1)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true_oh, self.iter)
        tprec = torch.mul(cross_ent, skel_true[:,1]).mean()
        tsens = torch.mul(cross_ent, skel_pred[:,1]).mean()
        cl_ce = (tprec+tsens)
        result = self.weight_ce * ce_loss + self.weight_clCE * cl_ce
        return result



