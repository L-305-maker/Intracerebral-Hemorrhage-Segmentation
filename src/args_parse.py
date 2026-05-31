import argparse

def args_parse():
    args = argparse.ArgumentParser(description="choose your parameters")

    args.add_argument("--epochs",type=int,default=30,help="the epochs the model run")
    args.add_argument("--model",choices=["baseline","Advanced_model"],default="baseline",help="choose the model you want to run")
    args.add_argument("--batch_size",type=int,default=8)
    args.add_argument("--lr",type=float,default=0.001)
    args.add_argument("--seed",type=int,default=42)
    args.add_argument("--image_size",type=int,default=128)
    args.add_argument("--base_channels",type=int,default=16)
    args.add_argument("--data_dir",type=str,default="brain_images")
    args.add_argument("--num_workers",type=int,default=0)
    args.add_argument("--save_dir",type=str,default="outputs")
    args.add_argument("--config_dir",type=str,default="configs")
    args.add_argument("--dropout",type=float,default=0)
    args.add_argument("--criterion",choices=["BCEWithLogitLoss","Weighted BCE+DiceLoss","Focal Loss","Tversky Loss"],default="BCEWithLogitLoss")
    args.add_argument("--Attention",action="store_true",default=False)
    args.add_argument("--BatchNorm",action="store_true",default=False)
    args.add_argument("--Double_Conv",action="store_true",default=False)
    args.add_argument("--mode",choices=["train","eval"],default="train")

    parsed_args = args.parse_args()
    return parsed_args
