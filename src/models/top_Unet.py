import torch
import torch.nn as nn
import torch.nn.functional as F


# ResNet: 带残差连接的卷积层
class Residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, dropout=0.0):
        super().__init__()
        bias = not use_batchnorm

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*layers)

        if in_channels != out_channels:
            shortcut = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)]
            if use_batchnorm:
                shortcut.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*shortcut)
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        output = self.conv(x)
        output = output + residual
        output = self.relu(output)
        return output


# Attention Gate: 过滤 skip connection 中的无关噪声
class Attention_gate(nn.Module):
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attention = self.relu(self.gate_conv(gate) + self.skip_conv(skip))
        attention = self.psi(attention)
        return skip * attention


# 轻量 ASPP: 增强 bottleneck 的多尺度感受野
class Light_ASPP(nn.Module):
    def __init__(self, channels, use_batchnorm=True):
        super().__init__()
        bias = not use_batchnorm
        branch_channels = max(channels // 4, 1)

        self.branch_one = nn.Conv2d(channels, branch_channels, kernel_size=1, padding=0, bias=bias)
        self.branch_two = nn.Conv2d(channels, branch_channels, kernel_size=3, padding=1, dilation=1, bias=bias)
        self.branch_three = nn.Conv2d(channels, branch_channels, kernel_size=3, padding=2, dilation=2, bias=bias)
        self.branch_four = nn.Conv2d(channels, branch_channels, kernel_size=3, padding=4, dilation=4, bias=bias)

        layers = [nn.Conv2d(branch_channels * 4, channels, kernel_size=1, padding=0, bias=bias)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU(inplace=True))
        self.project = nn.Sequential(*layers)

    def forward(self, x):
        output = torch.cat([
            self.branch_one(x),
            self.branch_two(x),
            self.branch_three(x),
            self.branch_four(x)
        ], dim=1)
        return self.project(output)


class Top_Unet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int,
                 use_batchnorm=True, use_residual=True, use_attention=True,
                 use_aspp=True, dropout=0.0):
        super().__init__()
        self.use_attention = use_attention

        a1 = base_channels
        a2 = base_channels * 2
        a3 = base_channels * 4
        a4 = base_channels * 8

        if use_residual:
            self.layer_one = Residual_block(in_channels, a1, use_batchnorm, dropout=0.0)
            self.layer_two = Residual_block(a1, a2, use_batchnorm, dropout=dropout * 0.5)
            self.layer_three = Residual_block(a2, a3, use_batchnorm, dropout=dropout)
            self.bottleneck = Residual_block(a3, a4, use_batchnorm, dropout=dropout)
            self.layer_four = Residual_block(a4 + a3, a3, use_batchnorm, dropout=dropout)
            self.layer_five = Residual_block(a3 + a2, a2, use_batchnorm, dropout=dropout * 0.5)
            self.layer_six = Residual_block(a2 + a1, a1, use_batchnorm, dropout=0.0)
        else:
            self.layer_one = self.make_conv_block(in_channels, a1, use_batchnorm, dropout=0.0)
            self.layer_two = self.make_conv_block(a1, a2, use_batchnorm, dropout=dropout * 0.5)
            self.layer_three = self.make_conv_block(a2, a3, use_batchnorm, dropout=dropout)
            self.bottleneck = self.make_conv_block(a3, a4, use_batchnorm, dropout=dropout)
            self.layer_four = self.make_conv_block(a4 + a3, a3, use_batchnorm, dropout=dropout)
            self.layer_five = self.make_conv_block(a3 + a2, a2, use_batchnorm, dropout=dropout * 0.5)
            self.layer_six = self.make_conv_block(a2 + a1, a1, use_batchnorm, dropout=0.0)

        self.pool_one = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_two = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_three = nn.MaxPool2d(kernel_size=2, stride=2)

        self.aspp = Light_ASPP(a4, use_batchnorm) if use_aspp else nn.Identity()

        if use_attention:
            self.att_three = Attention_gate(gate_channels=a4, skip_channels=a3, inter_channels=a3 // 2)
            self.att_two = Attention_gate(gate_channels=a3, skip_channels=a2, inter_channels=a2 // 2)
            self.att_one = Attention_gate(gate_channels=a2, skip_channels=a1, inter_channels=max(a1 // 2, 1))

        self.out_conv = nn.Conv2d(a1, out_channels, kernel_size=1)

    def make_conv_block(self, in_channels, out_channels, use_batchnorm=True, dropout=0.0):
        bias = not use_batchnorm
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        e1 = self.layer_one(x)
        e2 = self.layer_two(self.pool_one(e1))
        e3 = self.layer_three(self.pool_two(e2))

        b = self.bottleneck(self.pool_three(e3))
        b = self.aspp(b)

        d3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        skip3 = self.att_three(b, e3) if self.use_attention else e3
        d3 = torch.cat([d3, skip3], dim=1)
        d3 = self.layer_four(d3)

        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        skip2 = self.att_two(d3, e2) if self.use_attention else e2
        d2 = torch.cat([d2, skip2], dim=1)
        d2 = self.layer_five(d2)

        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        skip1 = self.att_one(d2, e1) if self.use_attention else e1
        d1 = torch.cat([d1, skip1], dim=1)
        d1 = self.layer_six(d1)

        return self.out_conv(d1)
