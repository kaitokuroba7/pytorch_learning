#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: BGD_Func.py
@time: 2021/3/30 19:32
"""
import numpy as np
import torch
import Function.utils as d2l
import torch.utils.data as Data
import time
from torch import nn, optim


def get_data_ch7():
    data = np.genfromtxt('./airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
           torch.tensor(data[:1500, -1], dtype=torch.float32)  # 前1500个样本，每个样本5个特征


def sgd(params, states, hyper_params):
    for p in params:
        p.data -= hyper_params['lr'] * p.grad.data


def train_ch7(optimizer_fn, states, hyper_params, features, labels, batch_size=10, num_epochs=2):
    # 初始化模型
    net, loss = d2l.linreg, d2l.squared_loss

    # w = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(features.shape[1], 1)), dtype=torch.float32),
    #                        requires_grad=True)
    #
    # b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
    w = torch.tensor(np.random.normal(0, 0.01, size=(features.shape[1], 1)), dtype=torch.float32, requires_grad=True)

    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    def eval_loss():
        return loss(net(features, w, b), labels).mean().item()

    ls = [eval_loss()]
    data_iter = Data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True
    )

    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            l = loss(net(X, w, b), y).mean()  # 使用平均损失

            # 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()

            l.backward()
            optimizer_fn([w, b], states, hyper_params)  # 迭代模型参数
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())  # 每100个样本记录下当前训练误差

    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')
    d2l.plt.show()


def train_sgd(lr, batch_size, features, labels, num_epoch=2):
    train_ch7(sgd, None, {'lr': lr}, features, labels, batch_size, num_epoch)


def train_pytorch_ch7(optimizer_fn, optimizer_hyper_params, features, labels, batch_size=10, num_epochs=2):
    # 初始化模型
    net = nn.Sequential(
        nn.Linear(features.shape)
    )


if __name__ == "__main__":
    pass
