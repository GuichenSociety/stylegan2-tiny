import math
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn



class MappingNetwork(nn.Module):
    def __init__(self, features, n_layers):
        super().__init__()

        layers = []
        for i in range(n_layers):

            layers.append(EqualizedLinear(features, features))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = nn.Sequential(*layers)


    def forward(self, z):
       z = F.normalize(z, dim=1)
       return self.net(z)


'''
    log_resolution 是表示有几层
    max_features   是最大的特征层数量
    n_features     是从哪个特征层数开始
'''
class Generator(nn.Module):
    def __init__(self, log_resolution, d_latent, n_features = 32, max_features = 512):
        super().__init__()

        #  [512, 512, 512, 256, 128, 64, 32]   log_resolution = 8
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)

        # 初始化参数
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))
        #
        self.style_block = StyleBlock(d_latent, features[0], features[0])
        self.to_rgb = ToRGB(d_latent, features[0])
        blocks = [GeneratorBlock(d_latent, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)
        self.up_sample = UpSample()

    def forward(self, w, input_noise):
        # w : (n_gen_blocks,batch_size,D_latent)
        # 获取特征向量的批次
        batch_size = w.shape[1]
        # 把输入的扩充批次
        x = self.initial_constant.expand(batch_size, -1, -1, -1)    # x : (batch_size,features[0],4,4)

        x = self.style_block(x, w[0], input_noise[0][1])
        rgb = self.to_rgb(x, w[0])
        for i in range(1, self.n_blocks):
            x = self.up_sample(x)
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            rgb = self.up_sample(rgb) + rgb_new
        return rgb


class GeneratorBlock(nn.Module):
    def __init__(self, d_latent, in_features, out_features):
        super().__init__()
        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)
        self.to_rgb = ToRGB(d_latent, out_features)

    def forward(self, x, w, noise):

        x = self.style_block1(x, w, noise[0])

        x = self.style_block2(x, w, noise[1])

        rgb = self.to_rgb(x, w)

        return x, rgb

class StyleBlock(nn.Module):
    def __init__(self, d_latent, in_features, out_features):
        super().__init__()

        # 把维度W（特征向量） 和 特征图进行关联
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        # 噪声
        self.scale_noise = nn.Parameter(torch.zeros(1))
        # 偏置
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w, noise):
        # w:(batch_size,d_latent)

        s = self.to_style(w)     # s:(batch_size,in_features)

        x = self.conv(x, s)      # x:(batch_size,out_features,h,w)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise

        return self.activation(x + self.bias[None, :, None, None])   # (batch_size,out_features,h,w)


class ToRGB(nn.Module):

    def __init__(self, d_latent, features):
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)
        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w):

        style = self.to_style(w)   # (batch_size,in_features)
        x = self.conv(x, style)    # (b,3,h,w)

        return self.activation(x + self.bias[None, :, None, None])    # (b,3,h,w)


class Conv2dWeightModulate(nn.Module):
    def __init__(self, in_features, out_features, kernel_size,demodulate= True, eps= 1e-8):
        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        # （stride = 1）  逆推导的，对分辨率不进行处理
        self.padding = (kernel_size - 1) // 2
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.eps = eps

    def forward(self, x, s):
        b, _, h, w = x.shape

        s = s[:, None, :, None, None]    # (batch_size,in_features)  -> (batch_size,1,in_features,1,1)   进行了填充

        weights = self.weight()[None, :, :, :, :]    # () -> (1,)    resize 一个维度
        weights = weights * s       # (batch_size,out_features,in_features,kernel_size,kernel_size)

        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv


        x = x.reshape(1, -1, h, w)    # x: (1,b*in_features,h,w)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)       # w: (b * self.out_features,in_features,kernel_size,kernel_size)
        x = F.conv2d(x, weights, padding=self.padding, groups=b)    # x:(1,b*out_features,h,w)
        return x.reshape(-1, self.out_features, h, w)               # x:(b,out_features,h,w)


class Discriminator(nn.Module):
    def __init__(self, log_resolution, n_features = 64, max_features = 512):
        super().__init__()
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
        n_blocks = len(features) - 1
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.std_dev = MiniBatchStdDev()
        final_features = features[-1] + 1
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    def forward(self, x):
        x = x - 0.5
        x = self.from_rgb(x)
        x = self.blocks(x)
        x = self.std_dev(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.final(x)     # (batch_size,1)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.residual = nn.Sequential(
            DownSample(),
            EqualizedConv2d(in_features, out_features, kernel_size=1)
        )

        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down_sample = DownSample()
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):

        residual = self.residual(x)
        x = self.block(x)
        x = self.down_sample(x)

        return (x + residual) * self.scale


class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size = 4):
        super().__init__()

        self.group_size = group_size


    def forward(self, x):

        assert x.shape[0] % self.group_size == 0

        grouped = x.view(self.group_size, -1)
        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        std = std.mean().view(1, 1, 1, 1)
        b, _, h, w = x.shape
        std = std.expand(b, -1, h, w)

        return torch.cat([x, std], dim=1)


class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = Smooth()

    def forward(self, x):
        x = self.smooth(x)
        return F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2), mode='bilinear', align_corners=False)


class UpSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.smooth = Smooth()

    def forward(self, x):
        return self.smooth(self.up_sample(x))


class Smooth(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]

        kernel = torch.tensor([[kernel]], dtype=torch.float)     # [1, 1, 3, 3]   [out,in,k,k]
        kernel /= kernel.sum()
        # 平滑核固定
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(-1, 1, h, w)
        x = self.pad(x)
        x = F.conv2d(x, self.kernel)
        return x.view(b, c, h, w)

# 学习速率均衡线性层
class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias = 0.):
        super().__init__()

        self.weight = EqualizedWeight([out_features, in_features])

        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x):

        return F.linear(x, self.weight(), bias=self.bias)

# 学习速率均衡 2D 卷积层
class EqualizedConv2d(nn.Module):
    def __init__(self, in_features, out_features,kernel_size, padding = 0):
        super().__init__()

        self.padding = padding

        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])

        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x):

        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


# 输入这个大小的权重参数[out_features, in_features, kernel_size, kernel_size] -> [out_features, in_features, kernel_size, kernel_size]
class EqualizedWeight(nn.Module):
    def __init__(self, shape):
        super().__init__()

        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):

        return self.weight * self.c


class GradientPenalty(nn.Module):

    def forward(self, x, d):
        batch_size = x.shape[0]

        gradients, *_ = torch.autograd.grad(outputs=d,
                                            inputs=x,
                                            grad_outputs=d.new_ones(d.shape),
                                            create_graph=True)

        gradients = gradients.reshape(batch_size, -1)
        norm = gradients.norm(2, dim=-1)
        return torch.mean(norm ** 2)


class PathLengthPenalty(nn.Module):
    def __init__(self, beta):
        super().__init__()

        self.beta = beta

        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)

        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w, x):

        device = x.device

        image_size = x.shape[2] * x.shape[3]

        y = torch.randn(x.shape, device=device)

        output = (x * y).sum() / math.sqrt(image_size)

        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs = w,
                                            grad_outputs = torch.ones(output.shape, device=device),
                                            create_graph = True)

        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        if self.steps > 0:
            a = self.exp_sum_a / (1 - self.beta ** self.steps)

            loss = torch.mean((norm - a) ** 2)
        else:

            loss = norm.new_tensor(0)

        mean = norm.mean().detach()

        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)

        self.steps.add_(1.)

        return loss


class Encoder(nn.Module):
    def __init__(self, size, w_dim=512):
        super().__init__()


    def forward(self, input):
        pass


if __name__ == '__main__':
    gen = Generator(8,512)

    print(gen)