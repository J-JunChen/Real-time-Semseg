# paper title: Deep Residual Learning for Image Recognition
# paper link: https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from typing import Type, Any, Callable, Union, List, Optional


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnet18_v1c', 'resnet50_v1c', 'resnet101_v1c']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',

    'resnet18_semseg': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth',
    'resnet50_semseg': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
    'resnet101_semseg': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
}

# if you have download the pretrained model in your local server,
# please change the link for yourself.
local_urls = {
    'resnet18': '/home/cjj/pretrained/resnet18-5c106cde.pth',
    'resnet34': '/home/cjj/pretrained/resnet34-333f7ec4.pth',
    'resnet50': '/home/cjj/pretrained/resnet50-19c8e357.pth',
    'resnet101': '/home/cjj/pretrained/resnet101-5d3b4d8f.pth',
    'resnet152': '/home/cjj/pretrained/resnet152-b121ed2d.pth',

    'resnet18_semseg': '/home/cjj/pretrained/resnet18-imagenet.pth',
    'resnet50_semseg': '/home/cjj/pretrained/resnet50-imagenet.pth',
    'resnet101_semseg': '/home/cjj/pretrained/resnet101-imagenet.pth'
}

def conv3x3(in_planes, out_planes, stride=1, dilation=1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1  # the input channel in residual block doesn't change.

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, norm_layer=None) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, norm_layer=None) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, 
                 block: Type[Union[BasicBlock, Bottleneck]], 
                 layers,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 num_classes=1000, 
                 deep_base=False, 
                 norm_layer=None) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.deep_base = deep_base   # this step is for Semseg
        if not self.deep_base:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
        else:
            self.inplanes = 128
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], previous_dilation=1 ,dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], previous_dilation=dilations[0], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], previous_dilation=dilations[1], dilation=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], previous_dilation=dilations[2], dilation=dilations[3])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, 
                    block: Type[Union[BasicBlock, Bottleneck]], 
                    planes, 
                    blocks, 
                    stride=1, 
                    previous_dilation=1, 
                    dilation=1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.deep_base:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, deep_base=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        if local_urls is None:
            model.load_state_dict(
                model_zoo.load_url(model_urls['resnet18']), strict=False)
        else:
            model.load_state_dict(torch.load(
                local_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=False, deep_base=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        if local_urls is None:
            model.load_state_dict(
                model_zoo.load_url(model_urls['resnet34']), strict=False)
        else:
            model.load_state_dict(torch.load(
                local_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=False, deep_base=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], deep_base=deep_base, **kwargs)
    if pretrained:
        if local_urls is None:
            model.load_state_dict(
                model_zoo.load_url(model_urls['resnet50']), strict=False)
        else:
            model.load_state_dict(torch.load(
                local_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, deep_base=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], deep_base=deep_base, **kwargs)
    if pretrained:
        if local_urls is None:
            model.load_state_dict(
                model_zoo.load_url(model_urls['resnet101']), strict=False)
        else:
            model.load_state_dict(torch.load(
                local_urls['resnet101']), strict=False)
    return model


def resnet152(pretrained=False, deep_base=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], deep_base=deep_base, **kwargs)
    if pretrained:
        if local_urls is None:
            model.load_state_dict(
                model_zoo.load_url(model_urls['resnet152']), strict=False)
        else:
            model.load_state_dict(torch.load(
                local_urls['resnet152']), strict=False)
    return model

def resnet18_semseg(pretrained=False, deep_base=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        if local_urls is None:
            model.load_state_dict(
                model_zoo.load_url(model_urls['resnet18_semseg']), strict=False)
        else:
            model.load_state_dict(torch.load(
                local_urls['resnet18_semseg'], map_location=torch.device('cpu')), strict=False)
    return model

def resnet50_semseg(pretrained=False, deep_base=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], deep_base=deep_base, **kwargs)
    if pretrained:
        if local_urls is None:
            model.load_state_dict(
                model_zoo.load_url(model_urls['resnet50_semseg']), strict=False)
        else:
            model.load_state_dict(torch.load(
                local_urls['resnet50_semseg'], map_location=torch.device('cpu')), strict=False)
    return model

def resnet101_semseg(pretrained=False, deep_base=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], deep_base=deep_base, **kwargs)
    if pretrained:
        if local_urls is None:
            model.load_state_dict(
                model_zoo.load_url(model_urls['resnet101_semseg']), strict=False)
        else:
            model.load_state_dict(torch.load(
                local_urls['resnet101_semseg'], map_location=torch.device('cpu')), strict=False)
    return model


if __name__ == "__main__":
    resnet = resnet50(pretrained=True, deep_base=True)
    x = torch.randn(1, 3, 224, 224)
    out = resnet(x)
    print(out.size())
    print(resnet)
