import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import random


def apply_activation(logits, activation=None, default='softmax'):

    act = activation if activation is not None else default
    
    if act == 'softmax':
        return F.softmax(logits, dim=-1)
    elif act == 'sigmoid':
        return torch.sigmoid(logits)
    elif act == 'log_softmax':
        return F.log_softmax(logits, dim=-1)
    elif act == 'none':
        return logits
    else:
        raise ValueError(f"Unknown activation: {act}")

# MSE Loss
class MyMSELoss(nn.Module):

    def __init__(self):
        super(MyMSELoss, self).__init__()

    def forward(self, logits, targets, activation=None):

        # apply softmax to logits
        predictions = apply_activation(logits, activation, default='softmax')

        squared_diff = (predictions - targets) ** 2

        return squared_diff.mean()


# MAE Loss
class MyMAELoss(nn.Module):

    def __init__(self):
        super(MyMAELoss, self).__init__()

    def forward(self, logits, targets, activation=None):

        # apply softmax to logits
        predictions = apply_activation(logits, activation, default='softmax')

        abs_diff = torch.abs(predictions - targets)

        return abs_diff.mean()


# KLDiv Loss
class MyKLDivLoss(nn.Module):

    def __init__(self, eps=1e-7):
        super(MyKLDivLoss, self).__init__()

        self.eps = eps

    def forward(self, logits, targets, activation=None):

        act = activation if activation is not None else 'softmax'

        # convert softmax to logsoftmax
        if act == 'softmax':
            log_predictions = F.log_softmax(logits, dim=-1)
        elif act == 'sigmoid':
            # workaround for sigmoid models
            log_predictions = torch.log(torch.sigmoid(logits) + self.eps)
        else:
            log_predictions = logits

        log_targets = torch.log(targets + self.eps)

        loss = targets * (log_targets - log_predictions)

        return loss.sum(dim=1).mean()


# JSD Loss
class MyJSDLoss(nn.Module):

    def __init__(self, eps=1e-7):
        super(MyJSDLoss, self).__init__()

        self.eps = eps

    def forward(self, logits, targets, activation=None):

        # apply softmax to logits
        predictions = F.softmax(logits, dim=-1)

        act = activation if activation is not None else 'softmax'
        if act == 'sigmoid':
             log_predictions = torch.log(predictions + self.eps)
        else:
             log_predictions = F.log_softmax(logits, dim=-1)

        m = 0.5 * (predictions + targets)

        log_m = torch.log(m + self.eps)
        log_targets = torch.log(targets + self.eps)

        kld_predictions_m = predictions * (log_predictions - log_m)
        kld_targets_m = targets * (log_targets - log_m)

        jsd_loss = 0.5 * (kld_predictions_m + kld_targets_m)

        return jsd_loss.sum(dim=1).mean()


# WBCE Loss
class MyWeightedBCELoss(nn.Module):

    def __init__(self, beta=1.0, eps=1e-7):
        super(MyWeightedBCELoss, self).__init__()

        self.beta = beta
        self.eps = eps

    def forward(self, logits, targets, activation=None):

        act = activation if activation is not None else 'softmax'
        
        if act == 'softmax':
            log_predictions = F.log_softmax(logits, dim=-1)
            predictions = torch.exp(log_predictions)
        elif act == 'sigmoid':
            predictions = torch.sigmoid(logits)
            log_predictions = torch.log(predictions + self.eps)

        pos_term = self.beta * targets * log_predictions

        neg_term = (1 - targets) * torch.log(1 - predictions + self.eps)

        loss = pos_term + neg_term

        return -loss.sum(dim=1).mean()


# Focal Loss
class MyFocalLoss(nn.Module):

    def __init__(self, alpha=0.5, gamma=1.0, eps=1e-7):
        super(MyFocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets, activation=None):

        predictions = apply_activation(logits, activation, default='softmax')
        
        act = activation if activation is not None else 'softmax'
        if act == 'sigmoid':
             log_predictions = torch.log(predictions + self.eps)
        else:
             log_predictions = F.log_softmax(logits, dim=-1)

        term_signal = self.alpha * targets * ((1 - predictions) ** self.gamma) * log_predictions

        term_background = (1 - self.alpha) * (1 - targets) * (predictions ** self.gamma) * torch.log(1 - predictions + self.eps)

        loss = - (term_signal + term_background)

        return loss.sum(dim=1).mean()


# Soft Dice Loss
class MySoftDiceLoss(nn.Module):
    def __init__(self, threshold=0.0005, eps=1e-7):
        super(MySoftDiceLoss, self).__init__()

        self.threshold = threshold
        self.eps = eps

    def forward(self, logits, targets, activation=None, eval_mode=False):

        predictions = apply_activation(logits, activation, default='sigmoid')

        if eval_mode:
            predictions = (predictions > self.threshold).float()

        target_mask = (targets > self.threshold).float()
        
        intersection = (predictions * target_mask).sum(dim=-1)
        union = predictions.sum(dim=-1) + target_mask.sum(dim=-1)
        
        dice = (2. * intersection + self.eps) / (union + self.eps)
        
        return 1.0 - dice.mean()


# Tversky-Loss
class MyTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, threshold=0.0005, eps=1e-7):
        super(MyTverskyLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.eps = eps

    def forward(self, logits, targets, activation=None, eval_mode=False):

        predictions = apply_activation(logits, activation, default='sigmoid')

        if eval_mode:
            predictions = (predictions > self.threshold).float()

        target_mask = (targets > self.threshold).float()
        
        tp = (predictions * target_mask).sum(dim=-1)
        
        fp = (predictions * (1.0 - target_mask)).sum(dim=-1)
        
        fn = ((1.0 - predictions) * target_mask).sum(dim=-1)
        
        tversky_index = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        
        return 1.0 - tversky_index.mean()


# Sliced Wasserstein Loss
class MySlicedEMDLoss(nn.Module):

    def __init__(self, side_length=60, n_projections=50):
        super(MySlicedEMDLoss, self).__init__()

        self.side_length = side_length
        self.n_projections = n_projections

        # create coordinate grid
        x = torch.linspace(-1, 1, side_length)
        y = torch.linspace(-1, 1, side_length)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')

        # flatten grid, shape [3600, 2]
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

        # create projection vectors
        theta = torch.linspace(0, np.pi, n_projections)
        directions = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)

        # project grid, shape [3600, n_projections]
        projections = torch.matmul(grid, directions.T)

        # sort indices (pre-calculated)
        sorted_indices = torch.argsort(projections, dim=0)

        # automatically move to GPU
        self.register_buffer('sorted_indices', sorted_indices)

    def forward(self, logits, targets, activation=None):

        # apply softmax to logits
        predictions = apply_activation(logits, activation, default='softmax')

        device = predictions.device
        sorted_indices = self.sorted_indices.to(device)

        batch_size = predictions.shape[0]

        # add batch dim to indices, shape [batch_size, 3600, n_projections]
        idx_expanded = sorted_indices.unsqueeze(0).expand(batch_size, -1, -1)

        # add batch dim to prediction and targets, shape [batch_size, 3600, n_projections]
        predictions_expanded = predictions.unsqueeze(-1).expand(-1, -1, self.n_projections)
        targets_expanded = targets.unsqueeze(-1).expand(-1, -1, self.n_projections)

        # project distribution (Radon Transformation)
        predictions_sorted = torch.gather(predictions_expanded, 1, idx_expanded)
        targets_sorted = torch.gather(targets_expanded, 1, idx_expanded)

        # calculate cdfs
        predictions_cdf = torch.cumsum(predictions_sorted, dim=1)
        targets_cdf = torch.cumsum(targets_sorted, dim=1)

        loss = torch.abs(predictions_cdf - targets_cdf).sum(dim=1).mean(dim=1).mean()

        return loss


# JSD + Wasserstein
class JSDWassersteinLoss(nn.Module):
    def __init__(self, jsd_weight=1.0, wasserstein_weight=1.0, eps=1e-7, side_length=60, n_projections=50):
        super(JSDWassersteinLoss, self).__init__()
        
        self.jsd_weight = jsd_weight
        self.wasserstein_weight = wasserstein_weight

        self.jsd_fn = MyJSDLoss(eps=eps)
        self.semd_fn = MySlicedEMDLoss(side_length=side_length, n_projections=n_projections)

    def forward(self, logits, targets, activation=None):
        
        loss_jsd = self.jsd_fn(logits, targets, activation=activation)
        
        loss_semd = self.semd_fn(logits, targets, activation=activation)
        
        total_loss = (self.jsd_weight * loss_jsd) + (self.wasserstein_weight * loss_semd)
        
        return total_loss
        

# MAE + Tversky
class MAETverskyLoss(nn.Module):

    def __init__(self, mae_weight=1.0, tversky_weight=1.0, threshold=0.0005, alpha=0.5, beta=0.5, eps=1e-7):
        super(MAETverskyLoss, self).__init__()
        
        self.mae_weight = mae_weight
        self.tversky_weight = tversky_weight

        self.mae_loss_fn = MyMAELoss() 
        self.tversky_loss_fn = MyTverskyLoss(threshold=threshold, alpha=alpha, beta=beta, eps=eps)

    def forward(self, logits, targets, activation=None, eval_mode=False):
        
        loss_mae = self.mae_loss_fn(logits, targets, activation)
        
        loss_tversky = self.tversky_loss_fn(logits, targets, activation=activation, eval_mode=eval_mode)
        
        total_loss = (self.mae_weight * loss_mae) + (self.tversky_weight * loss_tversky)
        
        return total_loss


# WBCE + Soft Dice
class WBCEDiceLoss(nn.Module):

    def __init__(self, wbce_weight=1.0, dice_weight=1.0, threshold=0.0005, beta=10.0, eps=1e-7):
        super(WBCEDiceLoss, self).__init__()
        
        self.wbce_weight = wbce_weight
        self.dice_weight = dice_weight

        self.wbce_loss_fn = MyWeightedBCELoss(beta=beta, eps=eps) 
        self.dice_loss_fn = MySoftDiceLoss(threshold=threshold, eps=eps)

    def forward(self, logits, targets, activation=None, eval_mode=False):
        
        loss_wbce = self.wbce_loss_fn(logits, targets, activation=activation)
        
        loss_dice = self.dice_loss_fn(logits, targets, activation=activation, eval_mode=eval_mode)
        
        total_loss = (self.wbce_weight * loss_wbce) + (self.dice_weight * loss_dice)
        
        return total_loss
