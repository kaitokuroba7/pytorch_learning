#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: Func_used.py
@time: 2021/3/15 15:07
"""
import torch
from torch import nn


def pool2d(X, pool_size, mode='max'):
    """
    forward calculate of pooling
    :param X:
    :param pool_size:
    :param mode:
    :return:
    """
    X = X.float()
    p_h, p_w = pool_size
    h, w = X.shape[0], X.shape[1]
    Y = torch.zeros(h - p_h + 1, w - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i+p_h, j:j+p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i+p_h, j:j+p_w].mean()

    return Y


if __name__ == "__main__":
    pass
