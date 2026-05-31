import random
import sys
from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.baseline import SimpleUNet
from src.models.Advanced_model import Advanced_Unet
from src.data_process import get_dataloaders
from src.args_parse import args_parse
from src.result_store import save_experiment_record
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits) #形状：Bx1xHxW，经过sigmoid将logits转换为概率值
    preds = (probs > 0.5).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))             #形状：B，计算每个分割与金标准的交集像素数
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))   #形状：B，计算每个分割和金标准的像素总数
    dice = (2.0 * intersection + eps) / (union + eps)               #形状：B，计算每个分割与金标准的交集占总数的比例
    return dice.mean()                                              #计算B个分割的平均Dice系数


def iou_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits) #形状：Bx1xHxW，经过sigmoid将logits转换为概率值
    preds = (probs > 0.5).float()
    targets = targets.float()   

    intersection = (preds * targets).sum(dim=(1, 2, 3))             #形状：B，计算每个分割与金标准的交集像素数 
    total = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))   #形状：B，计算每个分割和金标准的像素总数
    union = total - intersection                                    #形状：B，计算每个分割与金标准的并集像素数（总数减去交集）
    iou = (intersection + eps) / (union + eps)                      #形状：B，计算每个分割与金标准的交集占并集的比例
    return iou.mean()                                               #计算B个分割的平均IoU分数


def plot_images(data_loader):
    # Visualize a batch of images and masks from the dataloader
    import matplotlib.pyplot as plt
    images, masks = next(iter(data_loader))
    batch_size = images.size(0)
    fig, axes = plt.subplots(2, batch_size, figsize=(3 * batch_size, 6))
    if batch_size == 1:
        axes = np.array(axes).reshape(2, 1)

    for i in range(batch_size):
        axes[0, i].imshow(images[i, 0].cpu(), cmap="gray")
        axes[0, i].set_title("Image")
        axes[0, i].axis("off")

        axes[1, i].imshow(masks[i, 0].cpu(), cmap="gray")
        axes[1, i].set_title("Mask")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()

def train_epoch(model, loader, criterion, optimizer, epochs, device):
    
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    pbar = tqdm(loader, desc=f"训练 Epoch {epochs}", unit='batch') #tqdm用于显示进度条
    model.train()
    #============================================================================
    #在此完成一个训练周期的逻辑，包括前向传播、计算损失、反向传播和优化器更新等
    for i,(image_t,mask_t) in enumerate(pbar):
        images = image_t.to(device)
        mask = mask_t.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits,mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_coefficient(logits.detach(),mask).item()
        total_iou += iou_score(logits.detach(),mask).item()

        pbar.set_postfix({'loss':total_loss/(i+1),'Dice':total_dice/(i+1),"Iou":total_iou/(i+1)})
    #============================================================================

    return  {'loss': total_loss / (i+1), 'Dice': total_dice / (i+1), 'IoU': total_iou / (i+1)}


def valid_epoch( model: nn.Module, loader: DataLoader, criterion: nn.Module, epochs, device):
    
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    model.eval()
    #============================================================================
    #在此完成一个验证周期的逻辑，包括前向传播、计算损失等
    with torch.no_grad():
        pbar = tqdm(loader,desc=f"Epoch:{epochs}",unit='batch')
        for i,(images,mask) in enumerate(pbar):
            images = images.to(device)
            mask = mask.to(device)

            logits = model(images)
            loss = criterion(logits,mask)
            
            total_dice += dice_coefficient(logits,mask).item()
            total_iou += iou_score(logits,mask).item()
            total_loss += loss.item()

            pbar.set_postfix({'loss':total_loss/(i+1),'Dice':total_dice/(i+1),"Iou":total_iou/(i+1)})
    #============================================================================
    return  {'loss': total_loss / (i+1), 'Dice': total_dice / (i+1), 'IoU': total_iou / (i+1)}


def test(model: nn.Module, data_loader: DataLoader, out_dir: Path, save_dir, device):

    out_dir = Path(out_dir)
    save_dir = Path(save_dir)
    best_ckpt = save_dir / "best_model.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    else:
        print(f"Warning: Best model checkpoint not found at {best_ckpt}. ")
        return None

    total_dice = 0.0
    total_iou = 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    model.eval()
    batch_count = 0
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"测试", unit='batch')
        for i, (images, masks) in enumerate(pbar):
            batch_count = i + 1
            images = images.to(device)
            masks = masks.to(device)
            #============================================================================
            #在此完成一个测试周期的逻辑，包括前向传播、计算Dice和IoU等
            logits = model(images)
            
            total_dice += dice_coefficient(logits,masks).item()
            total_iou += iou_score(logits,masks).item()

            pbar.set_postfix({'Dice':total_dice/(i+1),"Iou":total_iou/(i+1)})

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5) * 255
            images = images.cpu().numpy() * 255
            masks = masks.cpu().numpy() * 255

            #============================================================================

            for j in range(len(images)):
                image = images[j, 0].astype(np.uint8) #原始图像
                masks_image  = masks[j, 0].astype(np.uint8)  #真实掩码
                pred  = preds[j, 0].astype(np.uint8)  #预测掩码
                Image.fromarray(pred).save(out_dir / f"{i:03d}_{j:03d}_pred.png")
                Image.fromarray(masks_image).save(out_dir / f"{i:03d}_{j:03d}_mask.png")
                Image.fromarray(image).save(out_dir / f"{i:03d}_{j:03d}_image.png")

    if batch_count == 0:
        return {"Dice": 0.0, "IoU": 0.0, "predictions_dir": str(out_dir)}
    return {
        "Dice": total_dice / batch_count,
        "IoU": total_iou / batch_count,
        "predictions_dir": str(out_dir),
    }

def train(model,optimizer,epochs,device,save_dir,train_loader,valid_loader,criterion):

    save_dir = Path(save_dir)
    model = model.to(device)

    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / "best_model.pth"

    best_dice = -1.0
    best_valid_metrics = None
    history = []
    for epoch in range(1, epochs + 1):
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, epoch, device)
        valid_metrics = valid_epoch(model,valid_loader,criterion, epoch, device)
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "valid": valid_metrics,
        })

        if valid_metrics['Dice'] > best_dice:
            best_dice = valid_metrics['Dice']
            best_valid_metrics = {"epoch": epoch, **valid_metrics}
            torch.save(model.state_dict(), best_ckpt)
            print(f"Saved best model to {best_ckpt} (Dice={best_dice:.4f})")

    print("Training finished.")
    print(f"Best validation Dice: {best_dice:.4f}")
    return {
        "best_valid": best_valid_metrics,
        "history": history,
        "best_model_path": str(best_ckpt),
    }


def main(args=None):
    args = args_parse() if args is None else args
    seed_everything(args.seed)
    
    if args.image_size % 4 != 0:
        raise ValueError("--image_size must be divisible by 4 for the current U-Net down/up sampling path.")
    

    if args.model == "baseline":
        model = SimpleUNet(in_channels=1, out_channels=1, base_channels=args.base_channels)
    elif args.model == "Advanced_model":
        model = Advanced_Unet(
            in_channels=1,
            out_channels=1,
            base_channels=args.base_channels,
            use_batchnorm=args.BatchNorm,
            use_double_conv=args.Double_Conv,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if args.criterion == "BCEWithLogitLoss":
        criterion = nn.BCEWithLogitsLoss()
    elif args.criterion == "Focal_Loss":
        from src.criterion.Focal_Loss import Focal_Loss
        criterion = Focal_Loss()
    elif args.criterion == "Tversky_Loss":
        from src.criterion.Tversky_Loss import Tversky_Loss
        criterion = Tversky_Loss()
    elif args.criterion == "Weighted_BCE+DiceLoss":
        from src.criterion.Weighted_DiceLoss import Weighted_DiceLoss
        criterion = Weighted_DiceLoss()
    else:
        raise ValueError(f"Unknown criterion: {args.criterion}")

    train_loader,valid_loader,test_loader = get_dataloaders(args.data_dir,args.seed,args.num_workers,args.image_size,args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    save_dir = Path(args.save_dir)
    train_result = None
    test_result = None

    if args.mode == "train":
        train_result = train(
            model,
            optimizer,
            args.epochs,
            device,
            save_dir,
            train_loader,
            valid_loader,
            criterion
        )
        test_result = test(
            model,
            test_loader,
            save_dir / "predictions",
            save_dir,
            device
        )
    elif args.mode == "eval":
        test_result = test(
            model,
            test_loader,
            save_dir / "predictions",
            save_dir,
            device
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    record_path = save_experiment_record(args, model, train_result, test_result)
    print(f"Saved experiment record to {record_path}")


if __name__ == "__main__":
    main()
