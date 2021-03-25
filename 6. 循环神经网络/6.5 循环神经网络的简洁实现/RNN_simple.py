#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: RNN_simple.py
@time: 2021/3/21 17:08
"""
import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import Function.utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

# 6.5.1 定义模型
num_hidden = 256
rnn_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hidden)

num_steps = 35
batch_size = 2
state = None

X = torch.rand(num_steps, batch_size, vocab_size)
Y, state_new = rnn_layer(X, state)   # Y表示经过一层RNN计算后的输出，state_new表示隐含层状态的输出

print(Y.shape, len(state_new), state_new[0].shape)

if __name__ == "__main__":
    pass
