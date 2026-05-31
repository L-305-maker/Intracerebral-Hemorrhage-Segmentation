import albumentations as A


def get_train_transforms(image_size=128):
    
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=1),
            A.Rotate(limit=15,border_mode=0,p=0.5),# 随机旋转 ±15°          
            A.HorizontalFlip(p=0.5), # 随机水平翻转            
            A.VerticalFlip(p=0.3),# 随机垂直翻转
            A.RandomScale(scale_limit=(-0.1, 0.1), interpolation=1,p=0.5), # 随机缩放 ±10%
            A.ShiftScaleRotate(shift_limit=0.1,scale_limit=0,rotate_limit=0,border_mode=0,p=0.5), # 随机平移 ±10%
            A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2,p=0.5),# 随机亮度±20% 和对比度±20%
            A.RandomGamma(gamma_limit=(80, 120), p=0.5), # gamma<1 图像变亮，gamma>1 图像变暗
            A.GaussNoise(var_limit=(10.0, 50.0),p=0.3),# 随机高斯噪声
            A.GaussianBlur(blur_limit=(3, 5),p=0.3),# 随机高斯模糊（模拟运动伪影或层厚导致的模糊）
            A.ElasticTransform(alpha=1,sigma=50,alpha_affine=50,p=0.3), # 弹性变形（模拟不同脑解剖结构差异）
            A.GridDistortion(num_steps=5,distort_limit=0.3,p=0.2), # 网格失真（模拟CT扫描时的不均匀性）
        ],
        additional_targets={'mask': 'mask'}
    )


def get_val_transforms(image_size=128):

    return A.Compose(
        [A.Resize(image_size, image_size, interpolation=1),],
        additional_targets={'mask': 'mask'}
    )