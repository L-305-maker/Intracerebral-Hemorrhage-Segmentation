import torch
import torch.nn as nn
import torch.nn.functional as F

class Focal_Loss(nn.Module):
    def __init__(self,alpha:float=0.75,gamma:float=2.0,reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self,logits,targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce)

        focal = self.alpha * (1 - pt) ** self.gamma * bce

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal