import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=True)


def NeuronNode():
    return neuron.LIFNode(surrogate_function=surrogate.ATan(), v_threshold=1.0, v_reset=0.0, detach_reset=True, decay_input=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.sn1 = NeuronNode()
        
        self.conv2 = conv3x3(planes, planes)
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

        out = out + identity
        # _ = self.testsn(out.detach())

        return out
    

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, T=4):
        super().__init__()

        self.inplanes = 128
        self.T = T

        self.conv1 = conv3x3(3, self.inplanes)
        self.sn1 = NeuronNode()
        
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        
        self.avgpool = layer.AvgPool2d(2)
        self.fc1 = layer.Linear(512 * block.expansion * 4 * 4, 256)
        self.sn2 = NeuronNode()
        # self.test2 = NeuronNode()
        self.fc2 = layer.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                NeuronNode()
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
       
        if len(x.shape) == 4:                       # [B, C, H, W]
            x = x.repeat(self.T, 1, 1, 1, 1)        # [T, B, C, H, W]
        elif len(x.shape) == 5:                     # [B, T, C, H, W]
            x = x.permute(1, 0, 2, 3, 4)            # [T, B, C, H, W]
        
        x = self.conv1(x)
        x = self.sn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        
        x = self.fc1(x)
        # _ = self.test2(x)
        x = self.sn2(x)
        x = self.fc2(x)

        return x.mean(dim=0)


def ResNet19(num_classes=10, T=4):
    model = ResNet(BasicBlock, [3, 3, 2], num_classes=num_classes, T=T)
    return model