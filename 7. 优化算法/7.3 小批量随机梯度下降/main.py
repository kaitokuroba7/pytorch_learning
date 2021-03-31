#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: main.py
@time: 2021/3/30 11:37
"""
import matplotlib
import numpy as np
import time
import torch
from torch import nn, optim
import sys
import Function.utils as d2l
import BGD_Func as Func

features, labels = Func.get_data_ch7()


print(features.shape)
print(labels.shape)

Func.train_sgd(lr=1, batch_size=1500, features=features, labels=labels, num_epoch=6)

Func.train_sgd(lr=0.005, batch_size=1, features=features, labels=labels, num_epoch=2)

Func.train_sgd(lr=0.05, batch_size=10, features=features, labels=labels, num_epoch=2)

Func.train_pytorch_ch7(optim.SGD, {"lr": 0.05}, features, labels, 10)


if __name__ == "__main__":
    pass
