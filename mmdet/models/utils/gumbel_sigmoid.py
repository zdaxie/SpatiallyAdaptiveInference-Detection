import torch
from torch import nn


class GumbelSigmoid(nn.Module):
    def __init__(self, max_T, decay_alpha):
        super(GumbelSigmoid, self).__init__()

        self.max_T = max_T
        self.decay_alpha = decay_alpha
        self.softmax = nn.Softmax(dim=1)
        self.p_value = 1e-8

        self.register_buffer('cur_T', torch.tensor(max_T))

    def forward(self, x):
        if self.training:
            _cur_T = self.cur_T
        else:
            _cur_T = 0.03

        # Shape <x> : [N, C, H, W]
        # Shape <r> : [N, C, H, W]
        r = 1 - x
        x = (x + self.p_value).log()
        r = (r + self.p_value).log()

        # Generate Noise
        x_N = torch.rand_like(x)
        r_N = torch.rand_like(r)
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()

        # Get Final Distribution
        x = x + x_N
        x = x / (_cur_T + self.p_value)
        r = r + r_N
        r = r / (_cur_T + self.p_value)

        x = torch.cat((x, r), dim=1)
        x = self.softmax(x)
        x = x[:, [0], :, :]

        if self.training:
            self.cur_T = self.cur_T * self.decay_alpha

        return x