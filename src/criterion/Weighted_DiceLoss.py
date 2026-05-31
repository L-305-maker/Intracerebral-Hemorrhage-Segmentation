import torch
import torch.nn as nn
import torch.nn.functional as F

class Weighted_DiceLoss(nn.Module):
    def __init__(self,bce_weight=0.5,smooth=1.0):
        super().__init__()
        self.weight = bce_weight
        self.smooth = smooth

    def forward(self,logits,targets):
        probs = torch.sigmoid(logits)

        intersection = (probs * targets).sum(dim=(1,2,3))
        denominator = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
        dice_loss = (2. * intersection + self.smooth) / (denominator + self.smooth)

        bce = F.binary_cross_entropy_with_logits(logits, targets)
        return self.weight * bce + (1 - self.weight) * (1 - dice_loss)