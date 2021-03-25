#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: ResNet_main.py
@time: 2021/3/24 14:55
"""
import torch
import ResNet_Func as Func
import torch.nn as nn
import Function.utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("use device: ", end='')
print(device)
blk = Func.Residual(3, 3)
print(blk)
X = torch.rand((4, 3, 6, 6))
print(blk(X).shape)

blk = Func.Residual(3, 6, use_1x1conv=True)  # 此时通道数量改变，需要使用1X1的卷积层
print(blk(X).shape)


# ResNet
# 5.11.2 ResNet模型
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

net.add_module("resnet_block1", Func.resnet_block(64, 64, 2, first_block=True))
net.add_module("resnet_block2", Func.resnet_block(64, 128, 2))
net.add_module("resnet_block3", Func.resnet_block(128, 256, 2))
net.add_module("resnet_block4", Func.resnet_block(256, 512, 2))

net.add_module("global_avg_pool", d2l.GlobalAvgPool2d())  # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(512, 10)))  # 四维到二维


X = torch.rand((1, 1, 224, 224)) # batch_size, in_channel, n_H, n_w
for name, layer in net.named_children():
    X = layer(X)
    print(name, ' output shape: \t', X.shape)

# 5.11.3 获取数据和训练模型
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

if __name__ == "__main__":
    pass
