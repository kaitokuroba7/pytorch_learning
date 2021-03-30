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


def get_data_ch7():
    data = np.genfromtxt('./airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
           torch.tensor(data[:1500, -1], dtype=torch.float32)  # 前1500个样本，每个样本5个特征


def sgd(params, states, hyper_params):
    for p in params:
        p.data -= hyper_params['lr'] * p.grad.data


def train_ch7(optimizer_fn, states, hyper_params, features, labels):
    pass

if __name__ == "__main__":
    pass
