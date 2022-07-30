import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
import torch_dct

from model.resnet_dct import ResNetDCT_2345, ResNetDCT_345

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ks,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out

class SpatialPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialPath, self).__init__()
        inner_channels = 64  # This figure from the original code
        # follow the resnet's the first three convolution
        self.downsample = nn.Sequential(
            ConvBNReLU(in_channels, inner_channels, 
                        ks=7, stride=2, padding=3),
            ConvBNReLU(inner_channels, inner_channels,
                       ks=3, stride=2, padding=1),
            ConvBNReLU(inner_channels, inner_channels,
                       ks=3, stride=2, padding=1)
        )
        # this conv 1x1 didn't appear in the paper
        self.conv1x1 = ConvBNReLU(
            inner_channels, out_channels, ks=1, stride=1, padding=0)

    def forward(self, x):
        x = self.downsample(x)
        out = self.conv1x1(x)
        return out

class ContextPath(nn.Module):
    def __init__(self, layers=50, block_size=8):
        super(ContextPath, self).__init__()
        assert layers in [18, 34, 50, 101]
        self.layers = layers
        self.block_size = block_size
        # Backbone
        # if layers in [18,34]:
        #     resnet = ResNetDCT_345(layers, in_channels=192, pretrained=False, deep_base=False, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
        #     # resnet = ResNetDCT_345(layers, pretrained=False, deep_base=False, strides=(1, 2, 1, 1), dilations=(1, 1, 2, 4))
        # self.layer2, self.layer3, self.layer4 = \
        #      resnet.layer2, resnet.layer3, resnet.layer4
        resnet = ResNetDCT_2345(layers, in_channels=192, pretrained=False, deep_base=False, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
        if layers in [18,34]:
            self.down_layer = resnet.down_layer
        self.layer1, self.layer2, self.layer3, self.layer4 = \
             resnet.layer1,resnet.layer2, resnet.layer3, resnet.layer4

    def forward(self, x):
        N, C, H, W = x.size()
        if self.layers in [18, 34]:
            x = self.down_layer(x)
        feat_4 = self.layer1(x)
        feat_8 = self.layer2(feat_4) 
        feat_16 = self.layer3(feat_8)
        feat_32 = self.layer4(feat_16)

        out = F.interpolate(feat_32, size=(H, W), mode='bilinear', \
            align_corners=True)
        # out = F.interpolate(out, scale_factor=self.block_size, mode='bilinear', \
        #     align_corners=True)
        return out


class Head(nn.Module):
    def __init__(self, in_channels, mid_channels, classes, dropout=0.1):
        super(Head, self).__init__()
        self.cls = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(mid_channels, classes, kernel_size=1)
        )

    def forward(self, x):
        out = self.cls(x)
        return out


class DCTNet(nn.Module):
    def __init__(self, 
                 layers=18, 
                 block_size=8, 
                 classes=2,  
                 vec_dim=300):
        super(DCTNet, self).__init__()
        self.block_size = block_size
        self.cp = ContextPath(layers, block_size)
        self.sp = SpatialPath(in_channels=3, out_channels=128)
        self.head = Head(in_channels=128+2048, mid_channels=256, classes=classes)
    
    def forward(self, x):
        feat_rgb = self.sp(x[0])
        feat_dct = self.cp(x[1])
        feat_fuse = torch.cat((feat_rgb, feat_dct), dim=1)
        logits = self.head(feat_fuse)
        out = F.interpolate(logits, scale_factor=self.block_size, mode='bilinear', \
            align_corners=True)
        return out

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    rgb = torch.rand(4, 3, 1024, 1024).cuda()
    dct = torch.rand(4, 192, 128, 128).cuda()
    input = [rgb, dct]
    model = DCTNet(layers=18, classes=5, vec_dim=100).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('DCTNet', output.size())
