#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: GRU_main.py
@time: 2021/3/28 11:00
"""
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import Function.utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()


num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)




if __name__ == "__main__":
    pass
