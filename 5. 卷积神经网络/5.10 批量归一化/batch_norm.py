#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: batch_norm.py
@time: 2021/3/17 10:35
"""
import time
import torch
import torch.nn.functional as F
import Function.utils as d2l
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    """

    :param is_training:
    :param X:
    :param gamma:
    :param beta:
    :param moving_mean:
    :param moving_var:
    :param eps:
    :param momentum:
    :return:
    """
    # 判断是训练模式还是预测模式
    if not is_training:
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        # 2维的全连接层和4维度的卷积网络
        if len(X.shape) == 2:
            # 使用全连接层的情况
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道上(axis=1)的均值和方差。这里我们需要保持
            # X的形状以便后面可以广播
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        #  训练模式下用当前的方差和均值做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1 - momentum) * mean
        moving_var = momentum * moving_var + (1 - momentum) * var

    Y = gamma * X_hat + beta
    return Y, moving_mean, moving_var


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.ones(shape))
        # 不参与求梯度和迭代的变量，全在内存上初始化为0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        # 如果X不在内存上, 将moving_mean和moving_var复制到X所在的现存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(self.training, X, self.gamma, self.beta, self.moving_mean,
                                                          self.moving_var, eps=1e-5, momentum=0.9)
        return Y


net = nn.Sequential(
    nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
    BatchNorm(num_features=6, num_dims=4),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(6, 16, 5),
    BatchNorm(num_features=16, num_dims=4),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),

    d2l.FlattenLayer(),
    nn.Linear(16 * 4 * 4, 120),
    BatchNorm(num_features=120, num_dims=2),
    nn.Sigmoid(),

    nn.Linear(120, 84),
    BatchNorm(num_features=84, num_dims=2),
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
