import argparse

def args_parse():
    args = argparse.ArgumentParser(description="choose your parameters")

    args.add_argument("--epochs",type=int,default=30,help="the epochs the model run")
    #epoch——迭代次数
    args.add_argument("--model",choices=["baseline","Advanced_Unet","cited_Unet","top_Unet"],default="baseline",help="choose the model you want to run")
    #model——模型名称
    args.add_argument("--batch_size",type=int,default=8)
    #batch_size每批大小
    args.add_argument("--lr",type=float,default=0.001)
    #学习率
    args.add_argument("--lr_scheduler",action="store_true",default=False)
    args.add_argument("--lr_patience",type=int,default=3)
    args.add_argument("--lr_factor",type=float,default=0.5)
    args.add_argument("--min_lr",type=float,default=1e-6)
    #用于学习率自适应调整
    args.add_argument("--seed",type=int,default=42)
    #固定随机种子
    args.add_argument("--image_size",type=int,default=128)
    #图像大小
    args.add_argument("--base_channels",type=int,default=16)
    args.add_argument("--data_dir",type=str,default="brain_images")
    args.add_argument("--augment",action="store_true",default=False)
    args.add_argument("--roi_crop",action="store_true",default=False)
    args.add_argument("--roi_threshold",type=int,default=5)
    args.add_argument("--roi_padding",type=int,default=8)
    args.add_argument("--num_workers",type=int,default=0)
    args.add_argument("--save_dir",type=str,default="outputs")
    args.add_argument("--config_dir",type=str,default="configs")
    args.add_argument("--dropout",type=float,default=0)
    args.add_argument("--threshold",type=float,default=0.45)
    #dropout正则化程度
    args.add_argument("--criterion",choices=["BCEWithLogitLoss","Weighted_BCE+DiceLoss","Focal_Loss","Tversky_Loss"],default="BCEWithLogitLoss")
    #选择损失函数
    args.add_argument("--Attention",action="store_true",default=False)
    args.add_argument("--BatchNorm",action="store_true",default=False)
    args.add_argument("--Double_Conv",action="store_true",default=False)
    args.add_argument("--Residual",action="store_true",default=False)
    args.add_argument("--ASPP",action="store_true",default=False)
    #开启模型中特定的层
    args.add_argument("--mode",choices=["train","eval"],default="train")
    #选择模式
    parsed_args = args.parse_args()
    return parsed_args
