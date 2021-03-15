#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: example2.py
@time: 2021/3/9 20:17
"""
# 4.1.2 Module子类
from torch import nn
import torch

X = torch.randn(2, 784)


class MySequential(nn.Module):

    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, inputs):

        for module in self._modules.values():
            inputs = module(inputs)
        return inputs


net = MySequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
print(net)
net(X)

if __name__ == "__main__":
    pass
