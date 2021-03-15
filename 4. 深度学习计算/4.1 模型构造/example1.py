#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: example1.py
@time: 2021/3/9 19:14
"""
import torch
from torch import nn
from collections import OrderedDict


class MLP(nn.Module):
    # 声明带有模型参数的层
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(784, 256)  # 隐含层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)  # 输出层

    # 定义前向计算
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


X = torch.randn(2, 784)
net = MLP()
print(net)
print(net(X))
X = X.detach().numpy().reshape(-1, 2)




if __name__ == "__main__":
    pass
