#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: test.py
@time: 2021/3/24 14:37
"""

import torch
import torch.nn as nn

in_channels, out_channels, kernel_size = 4, 4, 1
net = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
print(net)
X = torch.rand((7, 4, 6, 6))
Y = net(X)
print(Y.shape)

if __name__ == "__main__":
    pass
