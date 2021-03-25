#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: VGG.py
@time: 2021/3/16 15:55
"""
import VGG_Func as Func
import torch
import Function.utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
# 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
fc_features = 512 * 7 * 7  # c * w * h
fc_hidden_units = 4096  # 任意

net = Func.vgg(conv_arch, fc_features, fc_hidden_units)
X = torch.rand(1, 1, 224, 224)

for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)

# 5.7.3 获取训练数据和训练模型
ratio = 8
small_conv_arch = [(1, 1, 64 // ratio), (1, 64 // ratio, 128 // ratio), (2, 128 // ratio, 256 // ratio),
                   (2, 256 // ratio, 512 // ratio), (2, 512 // ratio, 512 // ratio)]
net = Func.vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
print(net)

batch_size = 64
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


if __name__ == "__main__":
    pass
