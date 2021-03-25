#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: Func_used.py
@time: 2021/3/16 14:22
"""
import torchvision
import torch
import torch.utils.data as Data


def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    """
    Download the fashion mnist dataset and then load into memory
    :param batch_size:
    :param resize:
    :param root:
    :return:
    """
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    train_iter = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_iter = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_iter, test_iter


if __name__ == "__main__":
    pass
