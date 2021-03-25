#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: RNN_Func.py
@time: 2021/3/19 18:40
"""
import RNN_basic as Func
import torch
import Function.utils as d2l
import torch.nn as nn
import time
import math


def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state, num_hiddens, vocab_size, device, idx_to_char,
                char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上一步的时间作为当前时间步的输入
        X = Func.to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))

    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn(rnn, get_params, init_rnn_states, num_inputs, num_hiddens, num_outputs, vocab_size, device,
                          corpus_indices, id_to_char, char_to_idx, is_random_iter, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params(num_inputs, num_hiddens, num_outputs, device)
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如果使用相邻采样,在epoch开始时初始化隐藏状态
            state = init_rnn_states(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_states(batch_size, num_hiddens, device)
            else:
                # 否则需要使用detach函数从计算图中分离隐藏状态，这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                for s in state:
                    s.detach_()

            inputs = Func.to_onehot(X, vocab_size)
            # output有num_step个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = Func.rnn(inputs, state, params)
            # 拼接之后的形状为(num_step*batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # Y的形状是(batch_size, num_steps), 转置后再变成长度为
            # batch * num_steps 的向量，这样和输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            # 使用交叉熵损失计算平均分类误差
            l = loss(outputs, y.long())

            # 梯度清零
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            Func.grad_clipping(params, clipping_theta, device)
            d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不必再做平均
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start
            ))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, Func.init_rnn_state,
                                        num_hiddens, vocab_size, device, Func.idx_to_char, char_to_idx))


if __name__ == "__main__":
    pass
