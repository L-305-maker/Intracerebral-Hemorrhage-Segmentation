import torch
import torch.nn as nn
import torch.nn.functional as F

class Tversky_Loss(nn.Module):
    def __init__(self,alpha=0.75,beta=0.5,smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self,logits,targets):
        probs = torch.sigmoid(logits)

        tp = (probs * targets).sum(dim=(1, 2, 3))
        fp = (probs * (1 - targets)).sum(dim=(1, 2, 3))
        fn = ((1 - probs) * targets).sum(dim=(1, 2, 3))

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky.mean()
