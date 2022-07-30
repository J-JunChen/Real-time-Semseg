# paper title: Pyramid scene parsing network (CVPR 2017)
# paper link: https://arxiv.org/abs/1612.01105
# reference code: https://github.com/hszhao/semseg
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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


class DCTNet_2345(nn.Module):
    def __init__(self, layers=50, dropout=0.1, classes=2):
        super(DCTNet_2345, self).__init__()
        assert layers in [18, 34, 50, 101]
        assert classes > 1
        self.layers = layers
        # Backbone
        resnet = ResNetDCT_2345(layers, pretrained=False, deep_base=False, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
        if layers in [18, 34]:
            self.down_layer = resnet.down_layer
        self.layer1, self.layer2, self.layer3, self.layer4 = \
             resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        
        if layers == 18 or layers == 34:
            fea_dim = 512
            aux_dim = 256
        else:
            fea_dim = 2048
            aux_dim = 1024
        down_dim = fea_dim // 4

        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, down_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(down_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(down_dim, classes, kernel_size=1)
        )

    def forward(self, x):
        N, C, H, W = x.size()
        if self.layers in [18, 34]:
            x =  self.down_layer(x)
        feat_4 = self.layer1(x)
        feat_8 = self.layer2(feat_4) 
        feat_16 = self.layer3(feat_8)
        feat_32 = self.layer4(feat_16)

        logits = self.cls(feat_32)
        out = F.interpolate(logits, size=(H, W), mode='bilinear',
                            align_corners=True)
       
        return out


class DCTNet_345(nn.Module):
    def __init__(self, layers=50, dropout=0.1, classes=2):
        super(DCTNet_345, self).__init__()
        assert layers in [18, 34, 50, 101]
        assert classes > 1
        # Backbone
        resnet = ResNetDCT_345(layers, pretrained=False, deep_base=False, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
        self.layer2, self.layer3, self.layer4 = \
             resnet.layer2, resnet.layer3, resnet.layer4
        
        if layers == 18 or layers == 34:
            fea_dim = 512
            aux_dim = 256
        else:
            fea_dim = 2048
            aux_dim = 1024
        down_dim = fea_dim // 4

        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, down_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(down_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(down_dim, classes, kernel_size=1)
        )
        
    def forward(self, x):
        N, C, H, W = x.size()
        # feat_4 = self.layer1(x)
        feat_8 = self.layer2(x) 
        feat_16 = self.layer3(feat_8)
        feat_32 = self.layer4(feat_16)

        logits = self.cls(feat_32)
        out = F.interpolate(logits, size=(H, W), mode='bilinear',
                            align_corners=True)
       
        return out


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # input = torch.rand(4, 3, 512, 512).cuda()
    input = torch.rand(4, 192, 128, 128).cuda()
    # input = torch.rand(4, 192, 64, 64).cuda()
    model = DCTNet_2345(layers=18,  dropout=0.1, classes=21).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('DCTNet', output.size())
