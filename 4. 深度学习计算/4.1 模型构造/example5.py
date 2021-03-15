#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: example5.py
@time: 2021/3/9 20:46
"""
import torch.nn as nn

net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU()
})


net['output'] == nn.Linear(256, 10)
print(net['linear'])
print(net.output)
print(net)

if __name__ == "__main__":
    pass
