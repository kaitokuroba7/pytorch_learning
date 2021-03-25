#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: RNN_main.py
@time: 2021/3/22 17:01
"""
import torch
import torch.nn as nn
import Function.utils as d2l
import RNN_model as Model
import RNN_Func as Func

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
num_steps = 35
num_hiddens = 256
# rnn_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens) # 已测试
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
model = Model.RNNModel(rnn_layer=rnn_layer, vocab_size=vocab_size).to(device)

res = Func.predict_rnn_pytorch('分开', 10, model, vocab_size=vocab_size, device=device,
                               idx_to_char=idx_to_char, char_to_idx=char_to_idx)

print(res)

if __name__ == "__main__":
    num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2  # 注意这里的学习率设置
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    Func.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                       corpus_indices, idx_to_char, char_to_idx,
                                       num_epochs, num_steps, lr, clipping_theta,
                                       batch_size, pred_period, pred_len, prefixes)
    pass
