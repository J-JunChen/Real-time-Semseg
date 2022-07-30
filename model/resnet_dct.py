import torch
from torch import Tensor
import torch.nn as nn
from model.resnet import resnet18, resnet34, resnet50, resnet101


__all__ = ['ResNetDCT_2345', 'ResNetDCT_345']

def init_weight(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class ResNetDCT_2345(nn.Module):
    """ has stage 2 ~ stage 5 of the ResNet """
    def __init__(self, 
                 layers,
                 in_channels=192,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 pretrained=False,
                 deep_base=False) -> None:
        super(ResNetDCT_2345, self).__init__()
        self.layers = layers
        if layers == 18:
            resnet = resnet18(pretrained, deep_base, strides=strides, dilations=dilations)
        elif layers == 34:
            resnet = resnet34(pretrained, deep_base, strides=strides, dilations=dilations)
        elif layers == 50:
            resnet = resnet50(pretrained, deep_base, strides=strides, dilations=dilations)
        elif layers == 101:
            resnet = resnet101(pretrained, deep_base, strides=strides, dilations=dilations)
        self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool, self.fc = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, resnet.avgpool, resnet.fc
        self.relu = nn.ReLU(inplace=True)
        if layers in [18, 34]:
            in_ch = self.layer1[0].conv1.in_channels
            self.down_layer = nn.Sequential(
                nn.Conv2d(in_channels, in_ch, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True)
            )
            # initialize the weight for only one layer
            for m in self.down_layer.modules():
                init_weight(m)
        else:
            out_ch = self.layer1[0].conv1.out_channels
            self.layer1[0].conv1 = nn.Conv2d(in_channels, out_ch, kernel_size=1, stride=1, bias=False)
            init_weight(self.layer1[0].conv1)
           
            out_ch = self.layer1[0].downsample[0].out_channels
            self.layer1[0].downsample[0] = nn.Conv2d(in_channels, out_ch, kernel_size=1, stride=1, bias=False)
            init_weight(self.layer1[0].downsample[0])

    def forward(self, x) -> Tensor:
        if self.layers in [18, 34]:
            x = self.down_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNetDCT_345(nn.Module):
    """ only has stage 3, stage 4, stage 5 of the ResNet """
    def __init__(self, 
                 layers,
                 in_channels=192,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 pretrained=False,
                 deep_base=False) -> None:
        super(ResNetDCT_345, self).__init__()
        if layers == 18:
            resnet = resnet18(pretrained, deep_base, strides=strides, dilations=dilations)
        elif layers == 34:
            resnet = resnet34(pretrained, deep_base, strides=strides, dilations=dilations)
        elif layers == 50:
            resnet = resnet50(pretrained, deep_base, strides=strides, dilations=dilations)
        elif layers == 101:
            resnet = resnet101(pretrained, deep_base, strides=strides, dilations=dilations)
        self.layer2, self.layer3, self.layer4, self.avgpool, self.fc = \
            resnet.layer2, resnet.layer3, resnet.layer4, resnet.avgpool, resnet.fc
        self.relu = nn.ReLU(inplace=True)

        out_ch = self.layer2[0].conv1.out_channels
        ks = self.layer2[0].conv1.kernel_size
        stride = self.layer2[0].conv1.stride
        padding =  self.layer2[0].conv1.padding
        self.layer2[0].conv1 = nn.Conv2d(in_channels, out_ch, kernel_size=ks, stride=stride, padding=padding, bias=False)
        init_weight(self.layer2[0].conv1)
        
        out_ch = self.layer2[0].downsample[0].out_channels
        self.layer2[0].downsample[0] = nn.Conv2d(in_channels, out_ch, kernel_size=1, stride=2, bias=False)
        init_weight(self.layer2[0].downsample[0])

    def forward(self, x) -> Tensor:
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    resnet = ResNetDCT_2345(layers=18, pretrained=False, deep_base=False)
    # x = torch.randn(10, 3, 224, 224)
    x = torch.randn(10, 192, 56, 56)
    # x = torch.randn(10, 192, 28, 28)
    out = resnet(x)
    print(resnet)
    print(out.size())
