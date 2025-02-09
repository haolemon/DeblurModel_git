import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from models.MyDeblurModel.model_utils import LayerNorm2d
from torch.nn import init
# from MyDeblur.model.NAFNet.NAFNet_arch import NAFBlock
from models.NAFNet.Baseline_arch import BaselineBlock
# from MLWNet.models.wavelet_block import LWN


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFInitBlock(nn.Module):
    def __init__(self, channel, DW_Expend=2, num_heads=4):
        super(NAFInitBlock, self).__init__()
        E_channel = channel * DW_Expend
        self.conv1 = nn.Conv2d(channel, E_channel, 1, 1, 0, groups=1, bias=True)
        self.conv2 = nn.Conv2d(E_channel, E_channel, 3, 1, 1, groups=E_channel, bias=True)
        self.conv3 = nn.Conv2d(E_channel // 2, E_channel // 2, 1, 1, 0, groups=1, bias=True)
        self.conv4 = nn.Conv2d(E_channel // 2, E_channel, 1, 1, 0, groups=1, bias=True)
        self.conv5 = nn.Conv2d(E_channel // 2, E_channel // 2, 1, 1, 0, groups=1, bias=True)

        self.sg = SimpleGate()
        self.norm1 = LayerNorm2d(channel)
        self.norm2 = LayerNorm2d(channel)
        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
        # self.act = nn.GELU()

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)

        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        return y + x * self.gamma

    def initialize(self):
        with torch.no_grad():
            num_residual_channels = min(self.conv2.in_channels, self.conv2.out_channels)
            for idx in range(num_residual_channels):
                self.conv2.weight[idx, :, 1, 1] += 1.


class Conv1x1(nn.Module):
    def __init__(self, channel, E_scale=2):
        super(Conv1x1, self).__init__()
        E_channel = channel * E_scale
        self.conv1 = nn.Conv2d(channel, E_channel, 1, 1, 0, groups=1, bias=True)
        self.conv2 = nn.Conv2d(E_channel, E_channel, 3, 1, 1, groups=E_channel, bias=True)
        self.conv3 = nn.Conv2d(E_channel // 2, E_channel // 2, 1, 1, 0, groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, 1, 1, 0)
        )
        self.norm = LayerNorm2d(channel)
        self.sg = SimpleGate()
        self.initialize()
        self.dropout = nn.Dropout(0.2)
        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x) * x
        x = self.conv3(x)

        return inp + self.beta * x

    def initialize(self):
        with torch.no_grad():
            num_residual_channels = min(self.conv2.in_channels, self.conv2.out_channels)
            for idx in range(num_residual_channels):
                self.conv2.weight[idx, :, 1, 1] += 1.


class Init_Conv(nn.Module):
    def __init__(self, channel, num_heads=16, E_scale=2):
        super(Init_Conv, self).__init__()
        E_channel = channel * E_scale
        self.conv1 = nn.Conv2d(channel, E_channel, 1, 1, 0, groups=1, bias=True)
        self.conv2 = nn.Conv2d(E_channel, E_channel, 3, 1, 1, groups=E_channel, bias=True)
        self.conv3 = nn.Conv2d(E_channel // 2, E_channel // 2, 1, 1, 0, groups=1, bias=True)

        # self.up = nn.PixelShuffle(2)
        self.norm = LayerNorm2d(channel)
        self.sg = SimpleGate()
        self.initialize()
        # self.dropout = nn.Dropout(0.2)
        # self.act = nn.Hardtanh(min_val=0., max_val=1.)
        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        # x = x.abs() * x

        return inp + self.beta * x

    def initialize(self):
        with torch.no_grad():
            num_residual_channels = min(self.conv2.in_channels, self.conv2.out_channels)
            for idx in range(num_residual_channels):
                self.conv2.weight[idx, :, 1, 1] += 1.


class Init_Conv_V2(nn.Module):
    def __init__(self, channel, num_heads, E_scale=2):
        super(Init_Conv_V2, self).__init__()
        E_channel = channel * E_scale
        # self.att = Attention(channel, num_heads)
        self.conv1 = nn.Conv2d(channel, E_channel, 1, 1, 0, groups=1, bias=True)
        self.conv2 = nn.Conv2d(E_channel, E_channel, 3, 1, 1, groups=E_channel, bias=True)
        self.conv3 = nn.Conv2d(E_channel // 2, E_channel // 2, 3, 1, 1, groups=E_channel // 2, bias=True)
        self.conv4 = nn.Conv2d(E_channel // 2, E_channel // 2, 1, 1, 0, groups=1, bias=True)

        self.norm1 = LayerNorm2d(channel)
        self.norm2 = LayerNorm2d(channel)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, 1, 1, 0)
        )
        self.sg = SimpleGate()
        self.initialize()
        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        # x = self.att(x)
        # y = inp + self.beta * x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x) * x

        x = self.conv3(x)
        x = self.conv4(x)

        return inp + self.gamma * x

    def initialize(self):
        with torch.no_grad():
            num_residual_channels = min(self.conv2.in_channels, self.conv2.out_channels)
            for idx in range(num_residual_channels):
                self.conv2.weight[idx, :, 1, 1] += 1.
            num_residual_channels = min(self.conv3.in_channels, self.conv3.out_channels)
            for idx in range(num_residual_channels):
                self.conv3.weight[idx, :, 1, 1] += 1.


class Init_Conv_V3(nn.Module):
    def __init__(self, channel, num_heads=16, E_scale=2):
        super(Init_Conv_V3, self).__init__()
        E_channel = channel * E_scale
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1, groups=1, bias=True)
        self.conv3 = nn.Conv2d(channel, channel, 3, 1, 1, groups=1, bias=True)

        # self.up = nn.PixelShuffle(2)
        self.norm = LayerNorm2d(channel)
        # self.sg = SimpleGate()
        # self.initialize()
        # self.dropout = nn.Dropout(0.2)
        self.act = nn.SiLU(inplace=True)
        # self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm(inp)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        # x = self.sg(x)
        x = self.conv3(x)
        sim_att = torch.sigmoid(x) - 0.5
        # x = x.abs() * x

        return (x + inp) * sim_att

    def initialize(self):
        with torch.no_grad():
            num_residual_channels = min(self.conv2.in_channels, self.conv2.out_channels)
            for idx in range(num_residual_channels):
                self.conv2.weight[idx, :, 1, 1] += 1.


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

    def initialize(self):
        with torch.no_grad():
            num_residual_channels = min(self.qkv_dwconv.in_channels, self.qkv_dwconv.out_channels)
            for idx in range(num_residual_channels):
                self.qkv_dwconv.weight[idx, :, 1, 1] += 1.


class Layer_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(Layer_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, dim, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # self.dw_q = nn.Conv2d(dim, dim, num_heads + 1, (1, num_heads), (num_heads // 2, 1), groups=dim, bias=True)
        # self.dw_k = nn.Conv2d(dim, dim, num_heads + 1, (1, num_heads), (num_heads // 2, 1), groups=dim, bias=True)
        self.q = nn.Conv2d(dim, dim, 1, (1, num_heads), 0, groups=1, bias=True)
        self.k = nn.Conv2d(dim, dim, 1, (1, num_heads), 0, groups=1, bias=True)
        self.dw_v = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, w, h = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q, k, v = self.q(q), self.k(k), self.dw_v(v)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # attn = rearrange(attn, 'b c w h -> b c (w h)')
        # attn = attn.softmax(dim=-1)
        # attn = rearrange(attn, 'b c (w h) -> b c w h', w=w, h=h)

        out = attn @ v

        out = self.project_out(out)
        return out

    def initialize(self):
        with torch.no_grad():
            num_residual_channels = min(self.dw_v.in_channels, self.dw_v.out_channels)
            for idx in range(num_residual_channels):
                self.dw_v.weight[idx, :, 1, 1] += 1.


class ResBlock(nn.Module):
    def __init__(self, channel, num_heads=16, E_scale=2):
        super(ResBlock, self).__init__()
        E_channel = channel * E_scale
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(channel, E_channel, 1, 1, 0, groups=1, bias=True)
        self.conv3 = nn.Conv2d(E_channel // 2, E_channel // 2, 1, 1, 0)

        # self.up = nn.PixelShuffle(2)
        self.norm = LayerNorm2d(channel)
        self.sg = SimpleGate()
        # self.initialize()
        # self.dropout = nn.Dropout(0.2)
        self.act = nn.GELU()
        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm(inp)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        # x = x.abs() * x

        return inp + self.beta * x


class ChannelAtt(nn.Module):
    def __init__(self, channel):
        super(ChannelAtt, self).__init__()
        self.dim = 1 / (channel ** 1/2)
        self.qkv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel * 3, 1, 1, 0)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inp):
        q, k, v = torch.chunk(self.qkv(inp).squeeze(-1), chunks=3, dim=1)
        att = q @ k.transpose(-1, -2)
        att = self.softmax(att * self.dim)
        return att @ v


class SP_block(nn.Module):
    def __init__(self, channel, num_heads=8):
        super(SP_block, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel // 2, 3, 1, 1, groups=1)
        self.conv2 = nn.Conv2d(channel * 32, channel * 32, 3, 1, 1, groups=channel * 32)
        self.conv3 = nn.Conv2d(channel * 32, channel * 32, 1, 1, 0, groups=1)
        self.conv4 = nn.Conv2d(channel * 32, channel * 32, 3, 1, 1, groups=channel * 32)
        self.conv5 = nn.Conv2d(channel // 2, channel, 3, 1, 1, groups=1)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel * 32, channel * 32, 1, 1, 0)
        )
        self.down = nn.PixelUnshuffle(8)
        self.up = nn.PixelShuffle(8)
        self.norm = LayerNorm2d(channel)
        self.act = nn.GELU()
        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm(inp)
        x = self.conv1(x)
        x = self.act(x)

        x = self.down(x)
        x = self.conv2(x)
        x = self.sca(x) * x
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.up(x)
        x = self.conv5(x)

        return inp + self.beta * x

    def initialize(self):
        with torch.no_grad():
            num_residual_channels = min(self.conv2.in_channels, self.conv2.out_channels)
            for idx in range(num_residual_channels):
                self.conv2.weight[idx, :, 1, 1] += 1.
                self.conv4.weight[idx, :, 1, 1] += 1


class ConvBn(nn.Module):
    def __init__(self, channel):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(channel, channel, 3, 1, 1)
        self.bn = LayerNorm2d(channel)
        self.act = nn.GELU()

    def forward(self, inp):
        x = self.bn(inp)
        x = self.conv(x)
        x = self.act(x)
        return inp + x


class UpBlock(nn.Module):
    def __init__(self, channel, scale=2):
        super().__init__()
        self.channel_up = nn.Conv2d(channel, channel * scale, 1, 1, 0, bias=False)
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.channel_up(x)
        x = self.up(x)
        return x


class Concat(nn.Module):
    def __init__(self, width):
        super(Concat, self).__init__()
        self.down1 = nn.Conv2d(width, width * 2, 3, 2, 1)
        self.down2 = nn.Conv2d(width * 4, width * 4, 3, 2, 1)
        self.down3 = nn.Conv2d(width * 8, width * 8, 3, 2, 1)

    def forward(self, x1, x2, x3):
        x1 = self.down1(x1)
        x2 = torch.cat([x1, x2], dim=1)
        x3 = torch.cat([self.down2(x2), x3], dim=1)
        return self.down3(x3)


if __name__ == '__main__':
    from torchinfo import summary

    w = 32
    d = 256
    model = SP_block(channel=w, num_heads=4)
    summary(model, (1, w, d, d), col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"))
