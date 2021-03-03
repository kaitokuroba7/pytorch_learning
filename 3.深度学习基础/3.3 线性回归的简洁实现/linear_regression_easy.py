#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: linear_regression_easy.py
@time: 2021/2/28 19:12
"""
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optim

# 3.3.1 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# 3.3.2 读取数据
batch_size = 10
# 将训练数据的特征和标签融合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)  # shuffle 的作用是打乱顺序


# for X, y in data_iter:
#     print(X, y)
#     break

# 3.3.3 定义模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # 定义前向传播网络
    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_inputs)
# print(net)
#
# for param in net.parameters():
#     print(param)

# 3.3.4 初始化模型参数
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# 3.3.5 定义损失函数
loss = nn.MSELoss()

# 3.3.6 定义优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

# 3.3.7 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()  # 梯度清零
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

if __name__ == "__main__":
    print(net.linear.weight, true_w)
    print(net.linear.bias, true_b)
    pass
