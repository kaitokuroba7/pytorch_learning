#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: Kaggle_house_main.py
@time: 2021/3/9 15:38
"""

import torch
import pandas as pd
from Kaggle_house_Func import k_fold, train_and_pred

print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)
# 读取数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(train_data.shape)
print(test_data.shape)
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 数据预处理
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# dummy_na = True将缺失值也当作合法的特征值并为其创建指标特征
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)


# 设置模型参数
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 6.0, 0, 32
num_hiddens1, num_hiddens2, drop_prob1, drop_prob2 = 128, 32, 0.2, 0.5
# train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
# print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))

if __name__ == "__main__":
    # train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size,
    #                num_hiddens1, num_hiddens2, drop_prob1, drop_prob2)
    train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
    pass
