#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: RNN_basic.py
@time: 2021/3/19 9:17
"""
import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import Function.utils as d2l

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()


# 6.4.1 one-hot 向量
def one_hot(x, n_class, dtype=torch.float32):
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


def to_onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


# 6.4.2 初始化模型参数


def get_params(num_inputs, num_hiddens, num_outputs, device):
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    # 隐含层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))
    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


def init_rnn_state(batch_size, num_hiddens, device):
    return torch.zeros((batch_size, num_hiddens), device=device),


def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state  # 上一个变量的隐含层
    outputs = []
    for X in inputs:
        H = torch.tanh((torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h))
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)

    return outputs, (H,)


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


if __name__ == "__main__":
    pass
