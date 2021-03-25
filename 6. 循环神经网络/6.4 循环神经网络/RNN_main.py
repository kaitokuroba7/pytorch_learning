#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: RNN_main.py
@time: 2021/3/19 11:29
"""
import torch
import Function.utils as d2l
import RNN_basic as Func
import RNN_Func as RNN_Func

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)

X = torch.arange(10).view(2, 5)
inputs = Func.to_onehot(X, vocab_size)
print(len(inputs), inputs[0].shape)

state = Func.init_rnn_state(X.shape[0], num_hiddens, device)
inputs = Func.to_onehot(X.to(device), vocab_size)
params = Func.get_params(num_inputs, num_hiddens, num_outputs, device)
output, state_new = Func.rnn(inputs, state, params)
print(len(output), output[0].shape, state_new[0].shape)

res = RNN_Func.predict_rnn('分开', 10, Func.rnn, params, Func.init_rnn_state, num_hiddens, vocab_size,
                           device, idx_to_char, char_to_idx)

print(res)
if __name__ == "__main__":
    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    RNN_Func.train_and_predict_rnn(Func.rnn, Func.get_params, Func.init_rnn_state, num_inputs, num_hiddens,
                                   num_outputs, vocab_size, device, corpus_indices, idx_to_char, char_to_idx, True,
                                   num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len,
                                   prefixes)

    pass
