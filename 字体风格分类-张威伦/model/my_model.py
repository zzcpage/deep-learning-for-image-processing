from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch.nn as nn
import torch
from torch import Tensor


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class HaarWaveletBlock(nn.Module):
    def __init__(self):
        super(HaarWaveletBlock, self).__init__()
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature_map_size = x.shape[1]
        x = torch.squeeze(self.global_avg_pooling(x))
        length = feature_map_size // 2
        temp = torch.reshape(x, (-1, length, 2))
        a = (temp[:, :, 0] + temp[:, :, 1]) / 2
        detail = (temp[:, :, 0] - temp[:, :, 1]) / 2
        length = length // 2
        while length != 16:  # 一级：32，acc：97.5， 二级：16，acc：97.875，三级：8, acc: 98.628, 四级：4，acc: 97.625, 五级：2，acc：97.5，六级：1，acc：97.375
            a = torch.reshape(a, (-1, length, 2))
            detail = torch.cat(((a[:, :, 0] - a[:, :, 1]) / 2, detail), dim=1)
            a = (a[:, :, 0] + a[:, :, 1]) / 2
            length = length // 2
        haar_info = torch.cat((a, detail), dim=1)
        # print('haar shape: {}'.format(haar_info.shape))
        return haar_info


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.relu_1 = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.relu_1(scale)
        return scale * x


class PowerNet(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 num_features: int = 1280,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2):
        super(PowerNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1))
        self.max_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.norm_1 = nn.BatchNorm2d(32)
        self.act_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32, stride=(1, 1), kernel_size=(5, 5))
        self.max_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.norm_2 = nn.BatchNorm2d(32)
        self.act_2 = nn.ReLU(inplace=True)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, stride=(1, 1), kernel_size=(5, 5))
        self.max_3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.norm_3 = nn.BatchNorm2d(64)
        self.act_3 = nn.ReLU(inplace=True)

        self.se = SqueezeExcitation(64)

        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=128, stride=(1, 1), kernel_size=(5, 5))
        self.max_4 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.norm_4 = nn.BatchNorm2d(128)
        self.act_4 = nn.ReLU(inplace=True)

        self.se_2 = SqueezeExcitation(128)

        self.har = HaarWaveletBlock()

        self.linear = nn.Linear(128, 4)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.max_1(x)
        x = self.norm_1(x)
        x = self.act_1(x)
        x = self.conv_2(x)
        x = self.max_2(x)
        x = self.norm_2(x)
        x = self.act_2(x)
        x = self.conv_3(x)
        x = self.max_3(x)
        x = self.norm_3(x)
        x = self.act_3(x)
        x = self.se(x)
        x = self.conv_4(x)
        x = self.max_4(x)
        x = self.norm_4(x)
        x = self.act_4(x)
        x = self.se_2(x)
        x = self.har(x)
        x = self.linear(x)
        return x


class SecondMerge(nn.Module):
    """
    二分枝网络
    """
    def __init__(self):
        super(SecondMerge, self).__init__()
        self.one = PowerNet()
        self.two = PowerNet()
        self.linear = nn.Linear(128, 4)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.one(x)
        x2 = self.two(x)
        y = torch.cat([x1, x2], dim=2)
        y = self.linear(y)
        return y