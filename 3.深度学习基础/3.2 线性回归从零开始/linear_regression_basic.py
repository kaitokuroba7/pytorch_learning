#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: linear_regression_basic.py
@time: 2021/2/27 20:06
"""

import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

# 3.2.1生成数据集
batch_size = 10
num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)

# 生成第二个特征和标签的散点图

plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)


# plt.show()


# 3.2.2读取数据
def data_iter(batch_size_func, features_func, labels_func):
    num_examples_func = len(features_func)
    indices = list(range(num_examples_func))
    random.shuffle(indices)
    for i in range(0, num_examples_func, batch_size_func):
        # num_examples_func代表了样本的总体的数量
        # batch_size_func代表了一个批次的数量
        j = torch.LongTensor(indices[i: min(i + batch_size_func, num_examples_func)])
        # 最后一次可能会不足
        yield features_func.index_select(0, j), labels_func.index_select(0, j)


for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

# 3.2.3 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# 3.2.4定义模型
def lin_reg(X_func, w_func, b_func):
    return torch.mm(X_func, w_func) + b_func
    # torch里面的矩阵相乘的算法


# 3.2.5定义损失函数
def squared_loss(y_hat, y_label):
    return (y_hat - y_label.view(y_hat.size())) ** 2 / 2


# 3.2.6定义优化算法
def sgd(params, lr_func, batch_size_func):
    for param in params:
        param.data -= lr_func * param.grad / batch_size_func


# 3.2.7训练模型
lr = 0.03
num_epochs = 3
net = lin_reg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        # 不要忘了梯度归零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print("epoch %d, loss %f" % (epoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)

if __name__ == "__main__":
    pass
