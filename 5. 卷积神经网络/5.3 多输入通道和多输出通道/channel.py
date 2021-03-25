#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: channel.py
@time: 2021/3/13 21:27
"""
import torch
from torch import nn
import Func_used as Func

if __name__ == "__main__":
    X = torch.tensor([
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ])

    K = torch.tensor([
        [[0, 1], [2, 3]],
        [[1, 2], [3, 4]]
    ])

    res = Func.corr2d_multi_in(X, K)
    print(res)

    K = torch.stack([K, K+1, K+2])
    print(K.shape)

    res = Func.corr2d_multi_in_out(X, K)
    print(res)
