import torch
import torch.nn as nn
import torch.nn.functional as F


class Advanced_Unet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int,use_batchnorm, use_double_conv, dropout ):
        super().__init__()
        bias = not use_batchnorm

        a1 = base_channels
        a2 = base_channels * 2
        a3 = base_channels * 4
        a4 = base_channels * 8


        layer_one = [nn.Conv2d(in_channels, a1, kernel_size=3, stride=1, padding=1, bias=bias)]
        if use_batchnorm:
            layer_one.append(nn.BatchNorm2d(a1))
        layer_one.append(nn.ReLU(inplace=True))
        if use_double_conv:
            layer_one.append(nn.Conv2d(a1, a1, kernel_size=3, stride=1, padding=1, bias=bias))
            if use_batchnorm:
                layer_one.append(nn.BatchNorm2d(a1))
            layer_one.append(nn.ReLU(inplace=True))
        self.layer_one = nn.Sequential(*layer_one)
        self.pool_one = nn.MaxPool2d(kernel_size=2, stride=2)


        layer_two = [nn.Conv2d(a1, a2, kernel_size=3, stride=1, padding=1, bias=bias)]
        if use_batchnorm:
            layer_two.append(nn.BatchNorm2d(a2))
        layer_two.append(nn.ReLU(inplace=True))
        if use_double_conv:
            layer_two.append(nn.Conv2d(a2, a2, kernel_size=3, stride=1, padding=1, bias=bias))
            if use_batchnorm:
                layer_two.append(nn.BatchNorm2d(a2))
            layer_two.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layer_two.append(nn.Dropout2d(dropout * 0.5))
        self.layer_two = nn.Sequential(*layer_two)
        self.pool_two = nn.MaxPool2d(kernel_size=2, stride=2)


        layer_three = [nn.Conv2d(a2, a3, kernel_size=3, stride=1, padding=1, bias=bias)]
        if use_batchnorm:
            layer_three.append(nn.BatchNorm2d(a3))
        layer_three.append(nn.ReLU(inplace=True))
        if use_double_conv:
            layer_three.append(nn.Conv2d(a3, a3, kernel_size=3, stride=1, padding=1, bias=bias))
            if use_batchnorm:
                layer_three.append(nn.BatchNorm2d(a3))
            layer_three.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layer_three.append(nn.Dropout2d(dropout))
        self.layer_three = nn.Sequential(*layer_three)
        self.pool_three = nn.MaxPool2d(kernel_size=2, stride=2)


        bottleneck = [nn.Conv2d(a3, a4, kernel_size=3, stride=1, padding=1, bias=bias)]
        if use_batchnorm:
            bottleneck.append(nn.BatchNorm2d(a4))
        bottleneck.append(nn.ReLU(inplace=True))
        if use_double_conv:
            bottleneck.append(nn.Conv2d(a4, a4, kernel_size=3, stride=1, padding=1, bias=bias))
            if use_batchnorm:
                bottleneck.append(nn.BatchNorm2d(a4))
            bottleneck.append(nn.ReLU(inplace=True))
        if dropout > 0:
            bottleneck.append(nn.Dropout2d(dropout))
        self.bottleneck = nn.Sequential(*bottleneck)


        layer_four = [nn.Conv2d(a4 + a3, a3, kernel_size=3, stride=1, padding=1, bias=bias)]
        if use_batchnorm:
            layer_four.append(nn.BatchNorm2d(a3))
        layer_four.append(nn.ReLU(inplace=True))
        if use_double_conv:
            layer_four.append(nn.Conv2d(a3, a3, kernel_size=3, stride=1, padding=1, bias=bias))
            if use_batchnorm:
                layer_four.append(nn.BatchNorm2d(a3))
            layer_four.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layer_four.append(nn.Dropout2d(dropout))
        self.layer_four = nn.Sequential(*layer_four)


        layer_five = [nn.Conv2d(a3 + a2, a2, kernel_size=3, stride=1, padding=1, bias=bias)]
        if use_batchnorm:
            layer_five.append(nn.BatchNorm2d(a2))
        layer_five.append(nn.ReLU(inplace=True))
        if use_double_conv:
            layer_five.append(nn.Conv2d(a2, a2, kernel_size=3, stride=1, padding=1, bias=bias))
            if use_batchnorm:
                layer_five.append(nn.BatchNorm2d(a2))
            layer_five.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layer_five.append(nn.Dropout2d(dropout * 0.5))
        self.layer_five = nn.Sequential(*layer_five)


        layer_six = [nn.Conv2d(a2 + a1, a1, kernel_size=3, stride=1, padding=1, bias=bias)]
        if use_batchnorm:
            layer_six.append(nn.BatchNorm2d(a1))
        layer_six.append(nn.ReLU(inplace=True))
        if use_double_conv:
            layer_six.append(nn.Conv2d(a1, a1, kernel_size=3, stride=1, padding=1, bias=bias))
            if use_batchnorm:
                layer_six.append(nn.BatchNorm2d(a1))
            layer_six.append(nn.ReLU(inplace=True))
        self.layer_six = nn.Sequential(*layer_six)


        self.out_conv = nn.Conv2d(a1, out_channels, kernel_size=1)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.layer_one(x)
        e2 = self.layer_two(self.pool_one(e1))
        e3 = self.layer_three(self.pool_two(e2))

        b = self.bottleneck(self.pool_three(e3))

        d3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.layer_four(d3)

        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.layer_five(d2)

        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.layer_six(d1)

        return self.out_conv(d1)
