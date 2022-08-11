import torch
import torch.nn as nn
from torch.nn import functional as F
import config
import math



# """手动定义卷积核(weight)和偏置"""
# w = torch.rand(16, 3, 5, 5)  # 16种3通道的5乘5卷积核
# b = torch.rand(16)  # 和卷积核种类数保持一致(不同通道共用一个bias)
#
# """定义输入样本"""
# x = torch.randn(1, 3, 28, 28)  # 1张3通道的28乘28的图像
#
# """2D卷积得到输出"""
# out = F.conv2d(x, w, b, stride=1, padding=1)  # 步长为1,外加1圈padding,即上下左右各补了1圈的0,
# print(out.shape)
#
# out = F.conv2d(x, w, b, stride=2, padding=2)  # 步长为2,外加2圈padding
# print(out.shape)
# out = F.conv2d(x, w)  # 步长为1,默认不padding, 不够的舍弃，所以对于28*28的图片来说，算完之后变成了24*24
# print(out.shape)
#
# kernel = [[1, 2, 1],
#           [2, 4, 2],
#           [1, 2, 1]]
#
# kernel = torch.tensor([[kernel]], dtype=torch.float)
# print(kernel.shape)
#
#
# # log_resolution = 8
# # a = [i for i in range(log_resolution - 2, -1, -1)]
# # print(a)
#
# log_resolution = 9
# max_features = 512
# n_features = 32
# features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
# print(features)
#
# s = torch.ones((4,16,4,4))
# s = s[:, None, :, None, None]
# print(s.shape)


# x = torch.ones((4,16,4,4))
# s = torch.ones((4,16,3,3))
# b, _, h, w = x.shape
# s = s[:, None, :, None, None]
# weights = torch.zeros(32,16,3,3)[None, :, :, :, :]
# weights = weights * s
# print(s.shape)
# print(weights.shape)
#
# _, _, *ws = weights.shape
# weights = weights.reshape(b * 32, *ws)
# print(*ws,weights)

# s = torch.randn((1, 1, 4, 4))
# initial_constant = nn.Parameter(s).expand(5, -1, -1, -1)
# print(s,initial_constant,initial_constant.shape)
#
#
# a = torch.rand((1)).item()
# print(a)

# batch_size = 16
# z2 = torch.randn(batch_size, config.D_latent)
# print(z2.shape)

# max_features = 512
# n_features = 64
# log_resolution = int(math.log2(config.Image_size))
# # features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
# features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
# print(features)

img_resolution = 512
img_resolution_log2 = int(math.log2(img_resolution))

block_resolutions = [2 ** i for i in range(img_resolution_log2 + 1)]
channels_dict = {res: min(32768 // res, 512) for res in block_resolutions}

fp16_resolution = max(2 ** (img_resolution_log2 + 1 - 4), 8)
print(fp16_resolution)