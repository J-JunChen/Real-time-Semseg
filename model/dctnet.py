import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
import torch.fft as fft

from model.resnet_dct import ResNetDCT_2345, ResNetDCT_345
from model.resnet import resnet18

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

        # resnet = ResNetDCT_2345(layers, in_channels=192, pretrained=False, deep_base=False, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
        # if layers in [18,34]:
        #     self.down_layer = resnet.down_layer
        resnet = resnet18(pretrained=False, deep_base=False, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = \
             resnet.layer1,resnet.layer2, resnet.layer3, resnet.layer4

    def forward(self, x):
        N, C, H, W = x.size()
        # if self.layers in [18, 34]:
        #     x = self.down_layer(x)
        x = self.layer0(x)
        feat_4 = self.layer1(x)
        feat_8 = self.layer2(feat_4) 
        feat_16 = self.layer3(feat_8)
        feat_32 = self.layer4(feat_16)

        # out = F.interpolate(feat_32, size=(H, W), mode='bilinear', \
        #     align_corners=True)
        out = F.interpolate(feat_32, scale_factor=4, mode='bilinear', \
            align_corners=True)
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
        self.head = Head(in_channels=640, mid_channels=512, classes=classes)
    
    def forward(self, x):
        # self.writer = writer
        freq_x = fft.fftn(x)
        freq_shift = fft.fftshift(freq_x)
        
        # low_freq_shift = self.easy_low_pass_filter(freq_x)
        # high_freq_shift = self.easy_high_pass_filter(freq_x)
        low_freq_shift, high_freq_shift = self.guassian_low_high_pass_filter(freq_shift)

        # low_freq_ishift = fft.ifftshift(low_freq_shift)
        high_freq_ishift = fft.ifftshift(high_freq_shift)
        
        # _low_freq_x = torch.abs(fft.ifftn(low_freq_ishift))
        _high_freq_x = torch.abs(fft.ifftn(high_freq_ishift))

        feat_rgb = self.sp(_high_freq_x)
        feat_dct = self.cp(x)
        feat_fuse = torch.cat((feat_rgb, feat_dct), dim=1)
        logits = self.head(feat_fuse)
        out = F.interpolate(logits, scale_factor=self.block_size, mode='bilinear', \
            align_corners=True)
        return out
    
    def guassian_low_high_pass_filter(self, x, D0=20):
        """reference code: https://blog.csdn.net/weixin_43959755/article/details/115528425 
        """
        _, _, H, W = x.size()
        y, z = torch.meshgrid(torch.arange(H), torch.arange(W))
        center = ((H-1)//2, (W-1)//2)
        dis_square = (y - center[0])**2 + (z - center[1])**2
        low_filter = torch.exp((-1) * dis_square / (2 * (D0 ** 2))).cuda()
        high_filter = 1 - low_filter
        return x * low_filter, x * high_filter


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    input = torch.rand(4, 3, 128, 128).cuda()
    # rgb = torch.rand(4, 3, 1024, 1024).cuda()
    # dct = torch.rand(4, 192, 128, 128).cuda()
    # input = [rgb, dct]
    model = DCTNet(layers=18, classes=5, vec_dim=100).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('DCTNet', output.size())
