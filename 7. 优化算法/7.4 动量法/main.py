#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: main.py
@time: 2021/3/31 20:50
"""
import Function.utils as d2l
import torch

eta = 0.4  # learning_rate


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2):  # 计算梯度
    return x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0


d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
d2l.plt.show()


eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
d2l.plt.show()

if __name__ == "__main__":
    pass
