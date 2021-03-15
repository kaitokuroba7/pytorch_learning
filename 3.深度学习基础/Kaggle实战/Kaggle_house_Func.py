#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: Kaggle_house_Func.py
@time: 2021/3/6 19:13
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import pandas as pd
import Function.utils as d2l

print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

# 3.16.4 训练模型
loss = torch.nn.MSELoss()


# def get_net(num_inputs, num_hiddens1, num_hiddens2, drop_prob1, drop_prob2):
#     net = nn.Sequential(
#         d2l.FlattenLayer(),
#         nn.Linear(num_inputs, num_hiddens1),
#         nn.ReLU(),
#         nn.Dropout(drop_prob1),
#         nn.Linear(num_hiddens1, num_hiddens2),
#         nn.ReLU(),
#         nn.Dropout(drop_prob2),
#         nn.Linear(num_hiddens2, 1)
#     )
#     for param in net.parameters():
#         nn.init.normal_(param, mean=0, std=0.01)
#     return net

def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net


def long_rmse(net, features, labels):
    with torch.no_grad():
        clipped_pred = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_pred.log(), labels.log()))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = Data.TensorDataset(train_features, train_labels)
    train_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
    # 这里使用了Adam 优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(long_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(long_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# 3.16.5 K折交叉验证
def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的数据
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    X_valid, y_valid = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


# 在K折交叉验证中我们训练K次并返回训练和验证的平均误差

def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    # net = get_net(train_features.shape[1], num_hiddens1, num_hiddens2, drop_prob1, drop_prob2)
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.DataFrame(preds.reshape(-1, 1))
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./submission.csv', index=False)


if __name__ == "__main__":
    pass
