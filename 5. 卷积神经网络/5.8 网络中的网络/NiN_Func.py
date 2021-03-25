#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: NiN_Func.py
@time: 2021/3/16 20:07
"""
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import Function.utils as Func

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )
    return blk





if __name__ == "__main__":
    pass
