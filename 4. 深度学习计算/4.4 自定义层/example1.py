#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: example1.py
@time: 2021/3/13 15:11
"""
import torch
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super(CenteredLayer, self).__init__()

    def forward(self, x):
        return x - x.mean()


if __name__ == "__main__":
    layer = CenteredLayer()
    output = layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))
    print(output)
    print("------------------这是一条分割线---------------------")
    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

    y = net(torch.rand(4, 8))
    print(y.mean().item())
