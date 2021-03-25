#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: example.py
@time: 2021/3/15 15:30
"""
import Func_used as Func
import torch
import torch.nn as nn

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
res = Func.pool2d(X, (2, 2))
print(res)

# 5.4.2 填充和步幅
X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
print(X)
pool2d = nn.MaxPool2d(3)
print(pool2d(X))

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
res = pool2d(X)
a = torch.tensor([[5, 7], [13, 15]], dtype=torch.float)
print(a == res)

# 指定非正方形窗口
pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
print(pool2d(X))

# 5.4.3 多通道、
X = torch.cat((X, X+1), dim=1)
print(X)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)

if __name__ == "__main__":
    pass
