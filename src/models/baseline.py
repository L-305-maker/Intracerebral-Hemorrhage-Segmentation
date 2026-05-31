import torch.nn as nn
import torch


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        #============================================================================
        #在此完成分割模型的定义
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels,c1,kernel_size=3,padding=1,bias=True),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(c1,c2,kernel_size=3,padding=1,bias=True),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(c2,c3,kernel_size=3,padding=1,bias=True),
            nn.ReLU(inplace=True)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c3,c3,kernel_size=3,padding=1,bias=True),
            nn.ReLU(inplace=True)
        )
        self.dec3_up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(c3+c2,c2,kernel_size=3,padding=1,bias=True),
            nn.ReLU(inplace=True)
        )
        self.dec2_up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(c2+c1,c1,kernel_size=3,padding=1,bias=True),
            nn.ReLU(inplace=True)
        )
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=3,padding=1,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1,out_channels,kernel_size=3,padding=1,bias=True)
        )
        #============================================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #============================================================================
        #在此完成分割模型的前向传播逻辑
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d3 = self.dec3_up(b)
        d3 = torch.cat([d3,e2],dim=1)
        d3 = self.dec3_conv(d3)

        d2 = self.dec2_up(d3)
        d2 = torch.cat([d2,e1],dim=1)
        d2 = self.dec2_conv(d2)

        d1 = self.dec1_conv(d2)
        #============================================================================

        return d1
