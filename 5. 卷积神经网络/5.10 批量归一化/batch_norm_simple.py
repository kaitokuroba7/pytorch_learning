#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: batch_norm_simple.py
@time: 2021/3/17 18:51
"""
import torch
import torch.nn as nn
import Function.utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = nn.Sequential(
    nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
    nn.BatchNorm2d(num_features=6),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(6, 16, 5),
    nn.BatchNorm2d(num_features=16),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),

    d2l.FlattenLayer(),
    nn.Linear(16 * 4 * 4, 120),
    nn.BatchNorm1d(num_features=120),
    nn.Sigmoid(),

    nn.Linear(120, 84),
    nn.BatchNorm1d(num_features=84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


if __name__ == "__main__":
    pass
