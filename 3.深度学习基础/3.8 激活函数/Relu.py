#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: Relu.py
@time: 2021/3/2 11:34
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import Function.utils as d2l


def xy_plot(x_val, y_val, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_val.detach().numpy(), y_val.detach().numpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')
    plt.show()


x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
xy_plot(x, y, 'relu')


y.sum().backward()
xy_plot(x, x.grad, 'grad of relu')

# 3.8.2.2 sigmoid 函数
y = x.sigmoid()
xy_plot(x, y, 'sigmoid')

x.grad.zero_()
y.sum().backward()
xy_plot(x, x.grad, 'grad of sigmoid')

# 3.8.2.3 tanh函数
y = x.tanh()
xy_plot(x, y, 'tanh')

x.grad.zero_()
y.sum().backward()
xy_plot(x, x.grad, 'grad of tanh')

if __name__ == "__main__":
    pass
