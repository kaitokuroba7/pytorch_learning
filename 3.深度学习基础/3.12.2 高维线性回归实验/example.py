#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: example.py
@time: 2021/3/3 17:23
"""
import torch
import torch.nn as nn
import numpy as np
import Function.utils as d2l
import torch.utils.data as Data

# 3.12.2 高维线性回归实验
# 这里定义p维度的数量为200
n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05

features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
train_features, test_features = features[: n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]


# 3.12.3 从零开始实现
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l2_penalty(w):
    return (w ** 2).sum() / 2


# 3.12.3.3定义训练和测试
batch_size, num_epochs, lr = 1, 100, 0.03
net, loss = d2l.linreg, d2l.squared_loss

dataset = Data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


def fit_and_plot(lambda_f):
    w, b = init_params()  # 初始化参数值
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            # 添加了L2 范数惩罚项
            l = loss(net(X, w, b), y) + lambda_f * l2_penalty(w)
            l = l.sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())

    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', w.norm().item())  # norm是指定范数的函数


if __name__ == "__main__":
    fit_and_plot(lambda_f=0)
    pass
