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

if __name__ == "__main__":
    pass
