import torch
import torch.nn as nn
from .BN_Free import *
from spikingjelly.activation_based import neuron, surrogate, layer



def NeurNode():
    return neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), init_tau=2.0, decay_input=True,
                                 v_threshold=1.0, v_reset=0.0, detach_reset=True)


def conv3x3(in_channels, out_channels, gamma=1.0):
    return nn.Sequential(
        Mul_ScaledWSConv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True, gamma=gamma),
        NeurNode()
    )


def conv1x1(in_channels, out_channels, gamma=1.0):
    return nn.Sequential(
        Mul_ScaledWSConv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=True, gamma=gamma),
        NeurNode()
    )


class SEWBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, alpha, beta):
        super(SEWBlock, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma_sn = 5.771
        
        self.conv1 = conv3x3(in_channels, mid_channels, gamma=self.beta)
        self.conv2 = conv3x3(mid_channels, in_channels, gamma=self.gamma_sn)

        # self.test_sn = NeurNode()

    def forward(self, x: torch.Tensor):
        
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        out = out*self.alpha + identity
        # _ = self.test_sn(out)

        return out


class ResNetN(nn.Module):
    def __init__(self, layer_list, num_classes, alpha=0.5):
        super(ResNetN, self).__init__()
        in_channels = 2
        conv = []
        self.gamma_sn = 5.771
        self.maxpool_var = 1.8
        
        expected_var = 1 / self.gamma_sn ** 2
        
        for cfg_dict in layer_list:
            
            beta = 1 / expected_var ** 0.5
            channels = cfg_dict['channels']

            if 'mid_channels' in cfg_dict:
                mid_channels = cfg_dict['mid_channels']
            else:
                mid_channels = channels

            if in_channels != channels:
                if cfg_dict['up_kernel_size'] == 3:
                    conv.append(conv3x3(in_channels, channels, gamma=1.3))
                elif cfg_dict['up_kernel_size'] == 1:
                    conv.append(conv1x1(in_channels, channels, gamma=1.3))
                else:
                    raise NotImplementedError

            in_channels = channels


            if 'num_blocks' in cfg_dict:
                num_blocks = cfg_dict['num_blocks']
                if cfg_dict['block_type'] == 'sew':
                    for _ in range(num_blocks):
                        conv.append(SEWBlock(in_channels, mid_channels, alpha=alpha, beta=beta))
                        expected_var += alpha ** 2 / self.gamma_sn ** 2
                else:
                    raise NotImplementedError
                
            if 'k_pool' in cfg_dict:
                k_pool = cfg_dict['k_pool']
                conv.append(layer.MaxPool2d(k_pool, k_pool))
                expected_var *= self.maxpool_var
                    
        self.conv = nn.Sequential(*conv)

        with torch.no_grad():
            x = torch.zeros([1, 1, 128, 128])
            for m in self.conv.modules():
                if isinstance(m, nn.MaxPool2d):
                    x = m(x)
            out_features = x.numel() * in_channels

        self.out = layer.Linear(out_features, num_classes)

    def forward(self, x: torch.Tensor):
        
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        x = self.conv(x)
        
        if self.out.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.out.step_mode == 'm':
            x = torch.flatten(x, 2)
            
        x = self.out(x)
        
        return x.mean(0)


def SEWResNet(alpha=0.5):
    layer_list = [
        {'channels': 32, 'up_kernel_size': 3, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
    ]
    num_classes = 11
    return ResNetN(layer_list, num_classes, alpha=alpha)