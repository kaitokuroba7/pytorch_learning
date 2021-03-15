#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: dropout.py
@time: 2021/3/6 14:30
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import Function.utils as d2l


def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把元素全部丢弃
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()

    return mask * X / keep_prob


num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens1, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, requires_grad=True)

params = [W1, b1, W2, b2, W3, b3]

# 3.13.2.2 定义模型
drop_prob1, drop_prob2 = 0.2, 0.5


def net(X_f, is_training=True):
    X_f = X_f.view(-1, num_inputs)
    H1 = (torch.matmul(X_f, W1) + b1).relu()
    if is_training:  # 只在训练模型中使用丢弃法
        H1 = dropout(H1, drop_prob1)
    H2 = (torch.matmul(H1, W2) + b2).relu()
    if is_training:
        H2 = dropout(H2, drop_prob2)
    return torch.matmul(H2, W3) + b3


def evaluate_accuracy(data_iter, net_f):
    acc_sum, n = 0.0, 0
    for X_f, y_f in data_iter:
        if isinstance(net_f, torch.nn.Module):
            net_f.eval()
            acc_sum += (net_f(X).argmax(dim=1) == y_f).float().sum().item()
            net_f.train()  # 改回训练模型
        else:
            if 'is_training' in net_f.__code__.co_varnames:
                acc_sum += (net(X, is_training=False).argmax(dim=1) == y_f).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1) == y_f).float().sum().item()
        n += y_f.shape[0]
    return acc_sum / n


# 3.13.2.3 训练和测试模型
num_epochs, lr, batch_size = 5, 100.0, 256
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

# 3.13.3 简洁实现
net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(drop_prob1),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(drop_prob2),
    nn.Linear(num_hiddens2, 10)
)

for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

# X = torch.arange(16).view(2, 8)
# res = dropout(X, 0.5)
# print(res)

if __name__ == "__main__":
    pass
