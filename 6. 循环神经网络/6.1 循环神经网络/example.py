#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: example.py
@time: 2021/3/17 20:32
"""
import torch

X, W_xh = torch.randn(3, 1), torch.randn(1, 4)
H, W_hh = torch.randn(3, 4), torch.randn(4, 4)

res = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)

print(res)

res_two = torch.matmul(torch.cat((X, H), dim=1), torch.cat((W_xh, W_hh), dim=0))

if __name__ == "__main__":
    pass
