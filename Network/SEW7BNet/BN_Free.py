import os
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from spikingjelly.activation_based import functional, base



class ScaledWSConv2d(nn.Conv2d):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
            bias=False, gamma=1.0, eps=1e-4):
        nn.Conv2d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
        self.gamma = gamma
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
        weight = self.gamma * (self.weight - mean) / (var * fan_in + self.eps) ** 0.5
        
        return weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class Mul_ScaledWSConv2d(ScaledWSConv2d, base.StepModule):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
            bias=False, gamma=1.0, eps=1e-4, step_mode: str = 's'
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                         groups, bias, gamma, eps)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 5:
                raise ValueError(f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


class ScaledWSLinear(nn.Linear):
    def __init__(
            self, in_features, out_features, bias=False, gamma=1.0, eps=1e-4):
        nn.Linear.__init__(
            self, in_features, out_features, bias)
        
        self.gamma = gamma
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1])
        mean = torch.mean(self.weight, axis=1, keepdims=True)
        var = torch.var(self.weight, axis=1, keepdims=True)
        weight = self.gamma * (self.weight - mean) / (var * fan_in + self.eps) ** 0.5
        
        return weight

    def forward(self, x):
        return F.linear(x, self.get_weight(), self.bias)


class Mul_ScaledWSLinear(ScaledWSLinear, base.StepModule):
    def __init__(
            self, in_features, out_features, bias=False, gamma=1.0, eps=1e-4, step_mode: str = 's'
    ) -> None:
        super().__init__(in_features, out_features, bias, gamma, eps)
        self.step_mode = step_mode


def BN_free_net_trams(Savemodel_path, net):

    if not os.path.exists(Savemodel_path):
        os.makedirs(Savemodel_path)

    for module in net.modules():
        if isinstance(module, (Mul_ScaledWSConv2d, Mul_ScaledWSLinear)):
            module.weight.data = module.get_weight()
            
    torch.save(net.state_dict(), Savemodel_path + "net_with_BN_Free.h5")