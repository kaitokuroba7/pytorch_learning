#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: mult_sensor.py
@time: 2021/3/2 14:42
"""
import torch
import numpy as np
import sys
import Function.utils as d2l

# 3.9.1 获取和读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 3.9.2 定义模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]

for param in params:
    param.requires_grad_(requires_grad=True)


# 3.9.3 定义激活函数
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))


# 3.9.4 定义模型
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2


# 3.9.5 定义损失函数
loss = torch.nn.CrossEntropyLoss()


# 3.9.6 训练模型
num_epochs, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)


if __name__ == "__main__":
    pass
