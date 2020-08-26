import numpy as np
import torch
import torch.nn as nn


class GaussianKernel(nn.Module):
    def __init__(self, size):
        super(GaussianKernel, self).__init__()

        s = (size - 1) // 2
        _x = torch.linspace(-s, s, size).reshape((size, 1)).repeat((1, size))
        _y = torch.linspace(-s, s, size).reshape((1, size)).repeat((size, 1))
        self.d = _x ** 2 + _y ** 2

    def forward(self, sigma):
        k = sigma ** 2
        A = k / (2. * np.pi)
        d = -k / 2. * self.d.cuda()
        B = torch.exp(d)
        B = A * B
        return B
