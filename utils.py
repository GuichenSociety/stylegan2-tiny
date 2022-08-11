import torch
import torch.nn as nn
from torch.nn import functional as F


class DiscriminatorLoss(nn.Module):
    def forward(self, f_real, f_fake):
        return F.relu(1 - f_real).mean(), F.relu(1 + f_fake).mean()


class GeneratorLoss(nn.Module):
    def forward(self, f_fake):
        return -f_fake.mean()