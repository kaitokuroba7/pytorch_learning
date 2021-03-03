#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: soft_max_from_zero.py
@time: 2021/3/1 16:57
"""

import torch
import torchvision
import numpy as np
import sys
import Function.utils as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 3.6.2 初始化模型参数
num_inputs = 28 * 28
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 3.6.3 实现softmax运算
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))
print(X.sum(dim=1, keepdim=True))


def softmax(X_f):
    X_exp = X_f.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(dim=1))


# 3.6.4 定义模型
def net(X_func):
    return softmax(torch.mm(X_func.view((-1, num_inputs)), W) + b)


# 3.6.5 定义损失函数
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))


def cross_entropy(h_hat, y_func):
    return - torch.log(h_hat.gather(1, y_func.view(-1, 1)))


# 3.6.6 计算分类准确率
def accuracy(y_hat_func, y_func):
    return (y_hat_func.argmax(dim=1) == y_func).float().mean().item()


print(accuracy(y_hat, y))


def evaluate_accuracy(data_iter, net_func):
    acc_sum, n = 0.0, 0
    for X_func, y_func in data_iter:
        acc_sum += (net_func(X_func).argmax(dim=1) == y_func).float().sum().item()
        n += y_func.shape[0]
    return acc_sum / n


print(evaluate_accuracy(test_iter, net))
# 3.6.7 训练模型
num_epochs, lr = 5, 0.1


def train_ch3(net_f, train_iter_f, test_iter_f, loss, num_epochs_f, batch_size_f,
              params=None, lr_f=None, optimizer=None):
    for epoch in range(num_epochs_f):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X_f, y_f in train_iter_f:
            y_hat_f = net_f(X_f)
            l = loss(y_hat_f, y_f).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr_f, batch_size_f)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat_f.argmax(dim=1) == y_f).sum().item()
            n += y_f.shape[0]

        test_acc = evaluate_accuracy(test_iter_f, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true+'\n'+pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])


if __name__ == "__main__":
    pass
