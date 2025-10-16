import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.jit import Final
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Union
from einops import rearrange, reduce
from ..modules.conv import Conv, DWConv,  RepConv, GhostConv, autopad
from ..modules.block import *


from ultralytics.utils.ops import make_divisible
from timm.layers import CondConv2d, trunc_normal_, use_fused_attn, to_2tuple
from timm.models import named_apply
_all__ = ['CGAFusion','C3k2_EFFconv','DCF']

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class SpatialAttention_CGA(nn.Module):
    def __init__(self):
        super(SpatialAttention_CGA, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention_CGA(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention_CGA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention_CGA(nn.Module):
    def __init__(self, dim):
        super(PixelAttention_CGA, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention_CGA()
        self.ca = ChannelAttention_CGA(dim, reduction)
        self.pa = PixelAttention_CGA(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, y = data
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result



class EFFconv(nn.Module):
    # Efficient Multi-Scale Conv Plus
    def __init__(self, channel=256, kernels=[1, 3, 5, 7]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // self.groups
        assert min_ch >= 16, f'channel must Greater than {16 * self.groups}, but {channel}'

        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch, c2=min_ch, k=ks))
        self.conv_1x1 = Conv(channel, channel, k=1)

    def forward(self, x):
        x_group = rearrange(x, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        x_convs = torch.stack([self.convs[i](x_group[..., i]) for i in range(len(self.convs))])
        x_convs = rearrange(x_convs, 'g bs ch h w -> bs (g ch) h w')
        x_convs = self.conv_1x1(x_convs)

        return x_convs

'''class EFFconv(nn.Module):  # Efficient Multi-Scale Conv Plus
    def __init__(self, channel=256, kernels=[3, 5, 7]):
        super().__init__()
        self.groups = len(kernels)

        # 确保通道数能被分组数整除
        if channel % self.groups != 0:
            # 
            channel = ((channel + self.groups - 1) // self.groups) * self.groups
            print(f"Warning: Adjusted channel from original to {channel} to be divisible by {self.groups}")

        min_ch = channel // self.groups
        assert min_ch >= 16, f'channel must Greater than {16 * self.groups}, but {channel}'

        # 如果输入通道数与调整后的不同，添加1x1卷积进行通道调整
        self.channel_adjust = None

        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch, c2=min_ch, k=ks))

        self.conv_1x1 = Conv(channel, channel, k=1)
        self.output_channel = channel

    def forward(self, x):
        # 
        original_channels = x.shape[1]
        target_channels = ((original_channels + self.groups - 1) // self.groups) * self.groups

        if original_channels != target_channels:
            #
            padding = target_channels - original_channels
            x = F.pad(x, (0, 0, 0, 0, 0, padding), mode='constant', value=0)

        x_group = rearrange(x, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        x_convs = torch.stack([self.convs[i](x_group[..., i]) for i in range(len(self.convs))])
        x_convs = rearrange(x_convs, 'g bs ch h w -> bs (g ch) h w')
        x_convs = self.conv_1x1(x_convs)

        # 如果输出通道数与原始不同，裁剪到原始大小
        if x_convs.shape[1] != original_channels and original_channels < x_convs.shape[1]:
            x_convs = x_convs[:, :original_channels, :, :]

        return x_convs'''

class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class Bottleneck_EFFconv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = EFFconv(c2)

class C3k_EFFconv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_EFFconv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_EFFconv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_EFFconv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_EFFconv(self.c, self.c, shortcut, g) for _ in
            range(n))



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class DCF(nn.Module):
    def __init__(
        self,
        inp: int,
    ) -> None:
        super(DCF, self).__init__()

        self.fuse1 = Conv(inp, inp, act=False)
        self.fuse2 = Conv(inp, inp, act=False)
        self.act = h_sigmoid()

    def forward(self, x):
        x_l, x_h = x
        B, C, H, W = x_l.shape
        inp = self.fuse1(x_l)
        sig_act = self.fuse2(x_h)
        sig_act = F.interpolate(self.act(sig_act), size=(H, W), mode='bilinear', align_corners=False)
        out = inp * sig_act
        return out