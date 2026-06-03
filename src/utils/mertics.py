import torch


def dice_coefficient(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.45,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.45,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    total = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    union = total - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()
