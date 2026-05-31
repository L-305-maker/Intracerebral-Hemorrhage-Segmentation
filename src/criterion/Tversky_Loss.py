import torch
import torch.nn as nn
import torch.nn.functional as F

class Tversky_Loss(nn.Module):
    #alpha控制惩罚FP，beta控制惩罚FN
    #本实验中对于脑出血的漏检风险更大，所以应该调大beta
    def __init__(self,alpha=0.3,beta=0.7,smooth=1.0):
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
