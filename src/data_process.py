from torch.utils.data import Dataset,DataLoader
import torch
from pathlib import Path
from typing import Tuple,List
from PIL import Image
import random
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

class ICHDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]], image_size: int = 256, augment: bool = False):
        super().__init__()
        self.image_size = image_size
        self.pairs = pairs
        self.augment = augment

    def transform(self, image: Image.Image, mask: Image.Image): #图像预处理和数据增强（如随机翻转、旋转等）
        #============================================================================
        #在此完成图像预处理和数据增强的逻辑
        target_size = (self.image_size, self.image_size)
        image = image.resize(target_size, resample=Image.Resampling.BILINEAR)
        mask = mask.resize(target_size, resample=Image.Resampling.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if random.random() > 0.5:
                image = image.transpose(Image.Transpose.ROTATE_90)
                mask = mask.transpose(Image.Transpose.ROTATE_90)

        image_t = torch.from_numpy(np.array(image, dtype=np.uint8)).unsqueeze(0).float() / 255.0
        mask_t  = torch.from_numpy(np.array(mask, dtype=np.uint8)).unsqueeze(0)
        mask_t = (mask_t > 0).float()
        #============================================================================
        return image_t, mask_t #形状：1xHxW, 1xHxW

    def __len__(self) -> int: #获取图像数目
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("L") #将图像转换为灰度模式
        mask = Image.open(mask_path).convert("L")
        image_t, mask_t = self.transform(image, mask)
        return image_t, mask_t #1xHxW, 1xHxW

def collect_pairs(data_dir: Path):
    img_paths = sorted(
        img_path
        for img_path in data_dir.glob("*_img.*")
        if img_path.suffix.lower() in IMAGE_EXTENSIONS
    )
    pairs: List[Tuple[Path, Path]] = []

    for img_path in img_paths:
        stem = img_path.stem
        prefix = stem[:-4] if stem.endswith("_img") else None
        if prefix is None:
            continue

        lab_candidates = sorted(
            lab_path
            for lab_path in data_dir.glob(f"{prefix}_lab.*")
            if lab_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not lab_candidates:
            continue

        pairs.append((img_path, lab_candidates[0]))
    return pairs

def get_dataloaders(data_dir,seed,num_workers,image_size,batch_size) -> Tuple[DataLoader, DataLoader, DataLoader]:
    pairs = collect_pairs(Path(data_dir))
    random.seed(seed)
    random.shuffle(pairs)

    # 按照7:2:1的比例划分训练集、验证集和测试集
    n = len(pairs)
    valid_count = max(1, int(n * 0.2))
    test_count = max(1, int(n * 0.1))
    train_count = n - valid_count - test_count
    if train_count < 1:
        raise RuntimeError("Not enough samples. Please add more data.")

    valid_pairs = pairs[:valid_count]
    test_pairs = pairs[valid_count:valid_count + test_count]
    train_pairs = pairs[valid_count + test_count:]

    #============================================================================
    #在此完成Dataset和DataLoader的定义
    train_ds = ICHDataset(train_pairs,image_size=image_size,augment=True)
    valid_ds = ICHDataset(valid_pairs,image_size=image_size,augment=False)
    test_ds = ICHDataset(test_pairs,image_size=image_size,augment=False)
    
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    valid_loader = DataLoader(
        valid_ds,
        shuffle=False,
        batch_size=batch_size,
        num_workers = num_workers,
        pin_memory=torch.cuda.is_available()
    )
    #============================================================================

    return train_loader, valid_loader, test_loader