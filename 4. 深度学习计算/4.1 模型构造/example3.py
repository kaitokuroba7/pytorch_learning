#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: example3.py
@time: 2021/3/9 20:21
"""
import torch.nn as nn
import torch

X = torch.randn(2, 784)
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10))
print(net[-1])
print(net)




if __name__ == "__main__":
    pass
