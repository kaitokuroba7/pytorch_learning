#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: LSTM_main.py
@time: 2021/3/29 21:15
"""
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import Function.utils as d2l
import LSTM_Func as Func

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print("will use", device)


def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))

    W_xi, W_hi, b_i = _three()  # 输入门参数
    W_xf, W_hf, b_f = _three()  # 遗忘门参数
    W_xo, W_ho, b_o = _three()  # 输出门参数
    W_xc, W_hc, b_c = _three()  # 候选记忆细胞参数

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q])


if __name__ == "__main__":
    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
    d2l.train_and_predict_rnn(Func.lstm, get_params, Func.init_lstm_state, num_hiddens,
                              vocab_size, device, corpus_indices, idx_to_char,
                              char_to_idx, False, num_epochs, num_steps, lr,
                              clipping_theta, batch_size, pred_period, pred_len,
                              prefixes)

    lr = 1e-2  # 注意调整学习率
    lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
    model = d2l.RNNModel(lstm_layer, vocab_size)
    d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                      corpus_indices, idx_to_char, char_to_idx,
                                      num_epochs, num_steps, lr, clipping_theta,
                                      batch_size, pred_period, pred_len, prefixes)
    pass
