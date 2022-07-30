# paper title: Pyramid scene parsing network (CVPR 2017)
# paper link: https://arxiv.org/abs/1612.01105
# reference code: https://github.com/hszhao/semseg
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import cv2
import torch_dct

import model.resnet as models

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


class TriSeNet1(nn.Module):
    def __init__(self, layers=50,  dropout=0.1, classes=2, fuse=8):
        super(TriSeNet1, self).__init__()
        assert layers in [18, 34, 50, 101]
        assert classes > 1
        self.fuse = fuse
        # Backbone
        if layers == 18:
            resnet = models.resnet18(pretrained=False, deep_base=False, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
        elif layers == 34:
            resnet = models.resnet34(pretrained=True, deep_base=False, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
        elif layers == 50:
            resnet = models.resnet50_semseg(pretrained=True, deep_base=True, strides=(1, 2, 1, 1), dilations=(1, 1, 2, 4))
        elif layers == 101:
            resnet = models.resnet101_semseg(pretrained=True, deep_base=True, strides=(1, 2, 1, 1), dilations=(1, 1, 2, 4))

        if layers == 18 or layers == 34:
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2,
                                        resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        
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
        if self.fuse == 16 or self.fuse == 8:
            self.fuse_16 = nn.Conv2d(fea_dim//2, classes, kernel_size=1)
        if self.fuse == 8:
            self.fuse_8 = nn.Conv2d(fea_dim//4, classes, kernel_size=1)
        # if self.training:
        #     self.aux = nn.Sequential(
        #         nn.Conv2d(aux_dim, aux_dim // 4, kernel_size=3, padding=1, bias=False),
        #         nn.BatchNorm2d(aux_dim // 4),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout2d(p=dropout),
        #         nn.Conv2d(aux_dim // 4, classes, kernel_size=1)
        #     )


    def forward(self, x):
        N, C, H, W = x.size()
        feat = self.layer0(x)
        feat_4 = self.layer1(feat)
        feat_8 = self.layer2(feat_4) # spatial information
        feat_16 = self.layer3(feat_8)
        feat_32 = self.layer4(feat_16)
        H32, W32 = feat_32.size()[2:]
       
        logits = self.cls(feat_32)
        if self.fuse == 16  or self.fuse == 8:
            logits_16 = self.fuse_16(feat_16)
            logits = F.interpolate(logits, scale_factor=2, mode='bilinear',
                                align_corners=True)
            logits += logits_16

        if self.fuse == 8:
            logits_8 = self.fuse_8(feat_8) 
            logits = F.interpolate(logits, scale_factor=2, mode='bilinear',
                                align_corners=True)
            logits += logits_8

        out = F.interpolate(logits, size=(H, W), mode='bilinear',
                            align_corners=True)
        # if self.training:
        #     aux = self.aux(feat_16)
        #     aux = F.interpolate(aux, size=(
        #         H, W), mode='bilinear', align_corners=True)
        #     return out, aux
        # else:  
        return out


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    input = torch.rand(4, 3, 512, 512).cuda()
    model = TriSeNet1(layers=18,  dropout=0.1, classes=21, fuse=16).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('FCNet', output.size())
