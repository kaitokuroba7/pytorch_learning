#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: NiN_main.py
@time: 2021/3/16 20:13
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import NiN_Func as Func
import Function.utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GlobalAvgPool2d(nn.Module):
    """
    全局的平均池化层可以通过将池化窗口设置成为输入的高和宽实现
    """

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


net = nn.Sequential(
    Func.nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    Func.nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    Func.nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    Func.nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    GlobalAvgPool2d(),
    # 将四维的输出转成二维的输出，其形状为(批量大小，10)
    d2l.FlattenLayer()
)

X = torch.rand(1, 1, 224, 224)
for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)

# 5.8.3 获取数据和训练模型
batch_size = 128
# 如出现"out of memory"的报错信息，可减少batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.002, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

if __name__ == "__main__":
    pass
