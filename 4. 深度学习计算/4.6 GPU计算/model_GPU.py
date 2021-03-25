#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: model_GPU.py
@time: 2021/3/13 16:52
"""
import torch
import torch.nn as nn

net = nn.Linear(3, 1)
output1 = list(net.parameters())[0].device
output2 = list(net.parameters())
print(output1)
print(output2)
net.cuda()
output3 = list(net.parameters())[0].device
print(output3)


if __name__ == "__main__":
    pass
