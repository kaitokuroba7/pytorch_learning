#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: GRU_Func.py
@time: 2021/3/28 11:08
"""
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size


def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True)
                )



if __name__ == "__main__":
    pass
