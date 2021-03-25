#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: test.py
@time: 2021/3/11 20:21
"""
import torch
from torch import nn

print(torch.cuda.is_available())

print(torch.cuda.device_count())

print(torch.cuda.current_device())

print(torch.cuda.get_device_name())


# 4.6.2 Tensor 的GPU计算
x = torch.tensor([1, 2, 3])
print(x)
x = x.cuda(0)
print(x)

if __name__ == "__main__":
    pass
