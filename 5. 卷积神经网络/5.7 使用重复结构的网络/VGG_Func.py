#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: VGG_Func.py
@time: 2021/3/16 15:37
"""
import time
import torch
from torch import nn, optim
import Function.utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)


def vgg(conv_arch, fc_features, fc_hidden_utils=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一个vgg_block都会使得高宽减半
        net.add_module('vgg_block_' + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    # 全连接层
    net.add_module("fc", nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(fc_features, fc_hidden_utils),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_utils, fc_hidden_utils),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_utils, 10)
    ))

    return net


if __name__ == "__main__":
    res = vgg_block(2, 32, 32)
    print(res)
    pass
