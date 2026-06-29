import torch
import torch.nn as nn
from .BN_Free import *
from spikingjelly.activation_based import neuron, surrogate, layer



def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
    return layer.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)


def Conv2d_BF(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, gamma=2.8467*1):
    return Mul_ScaledWSConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, gamma=gamma)


def Linear(in_planes, out_planes, bias=False):
    return layer.Linear(in_planes, out_planes, bias=bias)


def Linear_BF(in_planes, out_planes, bias=False, gamma=2.8467*1):
    return Mul_ScaledWSLinear(in_planes, out_planes, bias=bias, gamma=gamma)


def NeuronNode():
    return neuron.LIFNode(surrogate_function=surrogate.ATan(), v_threshold=1.0, v_reset=0.0, detach_reset=True, decay_input=False)



## VGG-11

class VGG_11_BF(nn.Module):
    def __init__(self, T=4, num_classes=10):
        super().__init__()
        self.T = T

        self.conv1 = Conv2d_BF(3, 64, kernel_size=3, padding=1, stride=1, bias=True, gamma=1.)    # 32 * 32
        self.sn1 = NeuronNode()

        self.conv2 = Conv2d_BF(64, 128, kernel_size=3, padding=1, stride=1, bias=True)  # 32 * 32
        self.sn2 = NeuronNode()                                              
        
        self.conv3 = Conv2d_BF(128, 256, kernel_size=3, padding=1, stride=2, bias=True) # 16 * 16
        self.sn3 = NeuronNode()
        
        self.conv4 = Conv2d_BF(256, 256, kernel_size=3, padding=1, stride=1, bias=True) # 16 * 16
        self.sn4 = NeuronNode()
        
        self.conv5 = Conv2d_BF(256, 512, kernel_size=3, padding=1, stride=2, bias=True) # 8 * 8
        self.sn5 = NeuronNode()

        self.conv6 = Conv2d_BF(512, 512, kernel_size=3, padding=1, stride=1, bias=True) # 8 * 8
        self.sn6 = NeuronNode()

        self.conv7 = Conv2d_BF(512, 512, kernel_size=3, padding=1, stride=1, bias=True) # 8 * 8
        self.sn7 = NeuronNode()

        self.conv8 = Conv2d_BF(512, 512, kernel_size=3, padding=1, stride=1, bias=True) # 8 * 8
        self.sn8 = NeuronNode()

        self.avgpool = layer.AvgPool2d(kernel_size=2, stride=2)

        self.linear1 = Linear_BF(4*4*512, 4096, bias=True)
        self.sn9 = NeuronNode()

        self.linear2 = Linear_BF(4096, 4096, bias=True)
        self.sn10 = NeuronNode()

        self.linear3 = Linear(4096, num_classes, bias=True)
        
        for m in self.modules():
            if isinstance(m, (Mul_ScaledWSConv2d, Mul_ScaledWSLinear)):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        if len(x.shape) == 4:                       # [B, C, H, W]
            x = x.repeat(self.T, 1, 1, 1, 1)        # [T, B, C, H, W]
        elif len(x.shape) == 5:                     # [B, T, C, H, W]
            x = x.permute(1, 0, 2, 3, 4)            # [T, B, C, H, W]

        x = self.sn1(self.conv1(x))
        x = self.sn2(self.conv2(x))
        x = self.sn3(self.conv3(x))
        x = self.sn4(self.conv4(x))
        x = self.sn5(self.conv5(x))
        x = self.sn6(self.conv6(x))
        x = self.sn7(self.conv7(x))
        x = self.sn8(self.conv8(x))
        x = self.avgpool(x)
        
        if self.linear1.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.linear1.step_mode == 'm':
            x = torch.flatten(x, 2)
            
        x = self.sn9(self.linear1(x))
        x = self.sn10(self.linear2(x))
        x = self.linear3(x)

        return x.mean(0)


class VGG_11_None(nn.Module):
    def __init__(self, T=4, num_classes=10):
        super().__init__()
        self.T = T

        self.conv1 = Conv2d(3, 64, kernel_size=3, padding=1, stride=1, bias=True)    # 32 * 32
        self.sn1 = NeuronNode()

        self.conv2 = Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=True)  # 32 * 32
        self.sn2 = NeuronNode()                                              
        
        self.conv3 = Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=True) # 16 * 16
        self.sn3 = NeuronNode()
        
        self.conv4 = Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=True) # 16 * 16
        self.sn4 = NeuronNode()
        
        self.conv5 = Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=True) # 8 * 8
        self.sn5 = NeuronNode()

        self.conv6 = Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=True) # 8 * 8
        self.sn6 = NeuronNode()

        self.conv7 = Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=True) # 8 * 8
        self.sn7 = NeuronNode()

        self.conv8 = Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=True) # 8 * 8
        self.sn8 = NeuronNode()

        self.avgpool = layer.AvgPool2d(kernel_size=2, stride=2)

        self.linear1 = Linear(4*4*512, 4096, bias=True)
        self.sn9 = NeuronNode()

        self.linear2 = Linear(4096, 4096, bias=True)
        self.sn10 = NeuronNode()

        self.linear3 = Linear(4096, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, (layer.Conv2d, layer.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        if len(x.shape) == 4:                       # [B, C, H, W]
            x = x.repeat(self.T, 1, 1, 1, 1)        # [T, B, C, H, W]
        elif len(x.shape) == 5:                     # [B, T, C, H, W]
            x = x.permute(1, 0, 2, 3, 4)            # [T, B, C, H, W]

        x = self.sn1(self.conv1(x))
        x = self.sn2(self.conv2(x))
        x = self.sn3(self.conv3(x))
        x = self.sn4(self.conv4(x))
        x = self.sn5(self.conv5(x))
        x = self.sn6(self.conv6(x))
        x = self.sn7(self.conv7(x))
        x = self.sn8(self.conv8(x))
        x = self.avgpool(x)
        
        if self.linear1.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.linear1.step_mode == 'm':
            x = torch.flatten(x, 2)
            
        x = self.sn9(self.linear1(x))
        x = self.sn10(self.linear2(x))
        x = self.linear3(x)

        return x.mean(0)


class VGG_11_BN(nn.Module):
    def __init__(self, T=4, num_classes=10):
        super().__init__()
        self.T = T

        self.conv1 = Conv2d(3, 64, kernel_size=3, padding=1, stride=1, bias=False)    # 32 * 32
        self.bn1 = layer.BatchNorm2d(64)
        self.sn1 = NeuronNode()

        self.conv2 = Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=False)  # 32 * 32
        self.bn2 = layer.BatchNorm2d(128)
        self.sn2 = NeuronNode()                                              
        
        self.conv3 = Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False) # 16 * 16
        self.bn3 = layer.BatchNorm2d(256)
        self.sn3 = NeuronNode()
        
        self.conv4 = Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False) # 16 * 16
        self.bn4 = layer.BatchNorm2d(256)
        self.sn4 = NeuronNode()
        
        self.conv5 = Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False) # 8 * 8
        self.bn5 = layer.BatchNorm2d(512)
        self.sn5 = NeuronNode()

        self.conv6 = Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False) # 8 * 8
        self.bn6 = layer.BatchNorm2d(512)
        self.sn6 = NeuronNode()

        self.conv7 = Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False) # 8 * 8
        self.bn7 = layer.BatchNorm2d(512)
        self.sn7 = NeuronNode()

        self.conv8 = Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False) # 8 * 8
        self.bn8 = layer.BatchNorm2d(512)
        self.sn8 = NeuronNode()

        self.avgpool = layer.AvgPool2d(kernel_size=2, stride=2)

        self.linear1 = Linear(4*4*512, 4096, bias=False)
        self.bn9 = layer.BatchNorm1d(4096)
        self.sn9 = NeuronNode()

        self.linear2 = Linear(4096, 4096, bias=False)
        self.bn10 = layer.BatchNorm1d(4096)
        self.sn10 = NeuronNode()

        self.linear3 = Linear(4096, num_classes, bias=True)
        
        for m in self.modules():
            if isinstance(m, (layer.Conv2d, layer.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if isinstance(m, (layer.BatchNorm1d, layer.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        if len(x.shape) == 4:                       # [B, C, H, W]
            x = x.repeat(self.T, 1, 1, 1, 1)        # [T, B, C, H, W]
        elif len(x.shape) == 5:                     # [B, T, C, H, W]
            x = x.permute(1, 0, 2, 3, 4)            # [T, B, C, H, W]

        x = self.sn1(self.bn1(self.conv1(x)))
        x = self.sn2(self.bn2(self.conv2(x)))
        x = self.sn3(self.bn3(self.conv3(x)))
        x = self.sn4(self.bn4(self.conv4(x)))
        x = self.sn5(self.bn5(self.conv5(x)))
        x = self.sn6(self.bn6(self.conv6(x)))
        x = self.sn7(self.bn7(self.conv7(x)))
        x = self.sn8(self.bn8(self.conv8(x)))
        x = self.avgpool(x)
        
        if self.linear1.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.linear1.step_mode == 'm':
            x = torch.flatten(x, 2)
            
        x = self.sn9(self.bn9(self.linear1(x).unsqueeze(3)).squeeze(3))
        x = self.sn10(self.bn10(self.linear2(x).unsqueeze(3)).squeeze(3))
        x = self.linear3(x)

        return x.mean(0)