import torch
import torch.nn as nn
from .BN_Free import *
from spikingjelly.activation_based import neuron, surrogate, layer



def conv3x3(in_planes, out_planes, stride=1, gamma=1.):
    """3x3 convolution with padding"""
    return Mul_ScaledWSConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True, gamma=gamma)


def conv1x1(in_planes, out_planes, stride=1, gamma=1.):
    """1x1 convolution"""
    return Mul_ScaledWSConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=True, gamma=gamma)


def NeuronNode():
    return neuron.LIFNode(surrogate_function=surrogate.ATan(), v_threshold=1.0, v_reset=0.0, detach_reset=True, decay_input=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, alpha, beta, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
            
        self.alpha = alpha
        self.beta = beta
        self.gamma_sn = 2.8467
        
        self.conv1 = conv3x3(inplanes, planes, stride, self.beta)
        self.sn1 = NeuronNode()
        
        self.conv2 = conv3x3(planes, planes, 1, self.gamma_sn)
        self.sn2 = NeuronNode()
        
        self.downsample = downsample
        # self.testsn = NeuronNode()

    def forward(self, x):
        
        identity = x

        out = self.conv1(x)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.sn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out*self.alpha + identity
        # _ = self.testsn(out.detach())

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, alpha, beta, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
            
        self.alpha = alpha
        self.beta = beta
        self.gamma_sn = 2.8467
        
        self.conv1 = conv1x1(inplanes, planes, 1, self.beta)
        self.sn1 = NeuronNode()
        
        self.conv2 = conv3x3(planes, planes, stride, self.gamma_sn)
        self.sn2 = NeuronNode()

        self.conv3 = conv1x1(planes, planes * self.expansion, 1, self.gamma_sn)
        self.sn3 = NeuronNode()
        
        self.downsample = downsample
        self.testsn = NeuronNode()

    def forward(self, x):
        
        identity = x

        out = self.conv1(x)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.sn2(out)

        out = self.conv3(out)
        out = self.sn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out*self.alpha + identity
        _ = self.testsn(out.detach())
        
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, imagenet=True, alpha=0.2, num_classes=1000, T=4):
        super().__init__()

        self.inplanes = 64
        self.T = T
        self.imagenet = imagenet
        self.gamma_sn = 2.8467
        
        if imagenet:
            self.conv1 = Mul_ScaledWSConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True, gamma=1.)
            self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = conv3x3(3, self.inplanes, gamma=1.)
            
        self.sn1 = NeuronNode()
        
        expected_var = 1 / self.gamma_sn ** 2
        self.layer1, expected_var = self._make_layer(block, 64, layers[0], alpha, expected_var)
        self.layer2, expected_var = self._make_layer(block, 128, layers[1], alpha, expected_var, stride=2)
        self.layer3, expected_var = self._make_layer(block, 256, layers[2], alpha, expected_var, stride=2)
        self.layer4, expected_var = self._make_layer(block, 512, layers[3], alpha, expected_var, stride=2)
        
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, Mul_ScaledWSConv2d):
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, alpha, expected_var, stride=1):
        
        beta = 1. / expected_var ** 0.5
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            expected_var += alpha ** 2 / self.gamma_sn ** 2
            beta = 1. / expected_var ** 0.5
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride, gamma=beta),
                NeuronNode()
            )
            # Reset expected var at a transition block
            expected_var = 1 / self.gamma_sn ** 2

        layers = []
        layers.append(block(self.inplanes, planes, alpha, beta, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            expected_var += alpha ** 2 / self.gamma_sn ** 2
            beta = 1. / expected_var ** 0.5
            layers.append(block(self.inplanes, planes, alpha, beta))
            
        return nn.Sequential(*layers), expected_var


    def forward(self, x):

        if len(x.shape) == 4:                       # [B, C, H, W]
            x = x.repeat(self.T, 1, 1, 1, 1)        # [T, B, C, H, W]
        elif len(x.shape) == 5:                     # [B, T, C, H, W]
            x = x.permute(1, 0, 2, 3, 4)            # [T, B, C, H, W]
            
        x = self.conv1(x)
        x = self.sn1(x)
                     
        if self.imagenet:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        
        x = self.fc(x)

        return x.mean(dim=0)

def ResNet18(num_classes=1000, imagenet=True, T=4, alpha=0.2):
    model = ResNet(BasicBlock, [2, 2, 2, 2], imagenet=imagenet, num_classes=num_classes, T=T, alpha=alpha)
    return model

def ResNet34(num_classes=1000, imagenet=True, T=4, alpha=0.2):
    model = ResNet(BasicBlock, [3, 4, 6, 3], imagenet=imagenet, num_classes=num_classes, T=T, alpha=alpha)
    return model

def ResNet50(num_classes=1000, imagenet=True, T=4, alpha=0.2):
    model = ResNet(Bottleneck, [3, 4, 6, 3], imagenet=imagenet, num_classes=num_classes, T=T, alpha=alpha)
    return model

def ResNet101(num_classes=1000, imagenet=True, T=4, alpha=0.2):
    model = ResNet(Bottleneck, [3, 4, 23, 3], imagenet=imagenet, num_classes=num_classes, T=T, alpha=alpha)
    return model

def ResNet152(num_classes=1000, imagenet=True, T=4, alpha=0.2):
    model = ResNet(Bottleneck, [3, 8, 36, 3], imagenet=imagenet, num_classes=num_classes, T=T, alpha=alpha)
    return model