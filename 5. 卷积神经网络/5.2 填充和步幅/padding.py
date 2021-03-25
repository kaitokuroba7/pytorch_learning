#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: padding.py
@time: 2021/3/13 21:01
"""
import torch
from torch import nn


def comp_conv2d(conv2d, X):
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    X = X.view((1, 1) + X.shape)  # 这里是四维了
    Y = conv2d(X)
    return Y.view(Y.shape[2:])


conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)


X = torch.rand(8, 8)
print(comp_conv2d(conv2d, X).shape)


# 使用高为5、宽度为3的卷积核。在高和宽两侧的填充分别为2和1
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)

if __name__ == "__main__":
    pass
