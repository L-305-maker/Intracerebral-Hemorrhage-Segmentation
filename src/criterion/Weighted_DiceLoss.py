import torch
import torch.nn as nn
import torch.nn.functional as F

class Weighted_DiceLoss(nn.Module):
    def __init__(self,weight=0.5,smooth=1.0):
        super().__init__()
        self.weight = weight
        self.smooth = smooth

    def forward(self,logits,targets):
        probs = torch.sigmoid(logits)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        bce = F.binary_cross_entropy_with_logits(logits, targets)
        return self.weight * bce + (1 - self.weight) * (1 - dice)