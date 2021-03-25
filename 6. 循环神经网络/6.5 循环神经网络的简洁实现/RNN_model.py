#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: RNN_model.py
@time: 2021/3/21 20:21
"""
import torch
import torch.nn as nn
import Function.utils as d2l


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state):  # inputs: (batch, seq_len)
        # 获取one-hot 向量表示
        X = d2l.to_onehot(inputs, self.vocab_size)  # X是个list
        Y, self.state = self.rnn(torch.stack(X), state)  # X = tensor(time_step, batch_size, vocab_size)
        # T = tensor(time_step, batch_size, hidden_size)
        # 全连接层首先会将Y的形状变成(num_steps * batch_size, num_hiddens), 它的输出
        # 形状为(num_step*batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state





if __name__ == "__main__":
    pass
