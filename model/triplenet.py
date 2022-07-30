# paper title: BisNet: Bilateral segmentation network for real-time semantic segmentation
# paper link: https://openaccess.thecvf.com/content_ECCV_2018/html/Changqian_Yu_BisNet_Bilateral_Segmentation_ECCV_2018_paper.html
# reference code: https://github.com/CoinCheung/BiSeNet
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch import nn
import torch.nn.functional as F
import math

import model.resnet as models

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3, stride=1, padding=1, with_relu=True):
        super(ConvBNReLU, self).__init__()
        self.with_relu = with_relu
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ks,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if self.with_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        out = self.bn(x)
        if self.with_relu:
            out = self.relu(out)
        return out

class UpModule(nn.Module):
    def __init__(self, in_channels, out_channels, up_scale=2):
        super(UpModule, self).__init__()
        self.up_scale=up_scale
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Upsample(scale_factor=up_scale, mode='nearest')
        )

    def forward(self, x):
        out = self.up(x)
        return out


class DownModule(nn.Module):
    def __init__(self, in_channels, out_channels, down_scale=2):
        super(DownModule, self).__init__()
        self.down = []
        for i in range(down_scale//2, 0, -1):
            if i == 1:
                self.down.append(ConvBNReLU(in_channels,out_channels,3,2,1,False))
            else:
                self.down.append(ConvBNReLU(in_channels,in_channels,3,2,1,True))
        self.down = nn.Sequential(*self.down)
        
    def forward(self, x):
        out = self.down(x)
        return out
    

class SelfAttentionBlock(nn.Module):
    """ Position attention module"""
    def __init__(self, in_dim):
        super(SelfAttentionBlock, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        proj_query = self.query_conv(y).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(y).view(m_batchsize, -1, width*height)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + y
        return out
        # return out


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super(ChannelAttentionModule, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels//reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        attention = self.channel_attention(x)
        y_ = torch.mul(y, attention)
        out = y_ + y
        return out


class SegHead(nn.Module):
    def __init__(self, in_channels, mid_channels, classes):
        super(SegHead, self).__init__()
        self.conv3x3_bn_relu = ConvBNReLU(in_channels=in_channels, out_channels=mid_channels,
                                          ks=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(
            in_channels=mid_channels, out_channels=classes, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        x = self.conv3x3_bn_relu(x)
        out = self.conv1x1(x)
        return out


class TriSeNet(nn.Module):
    def __init__(self, layers=18, classes=2):
        super(TriSeNet, self).__init__()

        if layers == 18:
            backbone = models.resnet18(pretrained=True, deep_base=False, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
        elif layers == 34:
            backbone = models.resnet34(pretrained=True, deep_base=False, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
        
        # the initial layer conv is 7x7, instead of three 3x3
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        # stage channels for resnet18 and resnet34 are:(64, 128, 256, 512)
        self.layer1, self.layer2, self.layer3, self.layer4 \
            = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        
        # self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling

        # self.up_16_8 = UpModule(in_channels=256, out_channels=128, up_scale=2)  # feat_16 up to the size of feat_8
        # self.up_32_8 = UpModule(in_channels=512, out_channels=128, up_scale=4)
        # self.down_8_16 = DownModule(in_channels=128, out_channels=256, down_scale=2) # feat_8 down to the size of feat_16
        # self.up_32_16 = UpModule(in_channels=512, out_channels=256, up_scale=2)
        self.down_8_32 = DownModule(in_channels=128, out_channels=512, down_scale=4)
        # self.down_16_32 = DownModule(in_channels=256, out_channels=512, down_scale=2)
        self.relu = nn.ReLU(inplace=True)

        self.sa_8_32 = SelfAttentionBlock(512)
        # self.ca_32_8 = ChannelAttentionModule(in_channels=128, reduction=4)
        
        # self.cp = ContextPath(in_channels=3, out_channels=128, backbone=resnet)
        
        self.seg_head = SegHead(in_channels=640, mid_channels=256, classes=classes)

    def forward(self, x):
        N, C, H, W = x.size()
        feat_4 = self.layer0(x)
        feat_4 = self.layer1(feat_4)
        feat_8 = self.layer2(feat_4)  # 1/8 of spatial size
        feat_16 = self.layer3(feat_8)  # 1/16 of spatial size
        feat_32 = self.layer4(feat_16)  # 1/32 of spatial size
        H8, W8 = feat_8.size()[2:]
        H16, W16 = feat_16.size()[2:]
        H32, W32 = feat_32.size()[2:]
        
        # gap_feat = self.gap(feat_32)

        # feat_16_8 = self.up_16_8(feat_16)
        # feat_32_8 = self.up_32_8(feat_32)
        # feat_8_16 = self.down_8_16(feat_8)
        # feat_32_16 = self.up_32_16(feat_32)
        feat_8_32 = self.down_8_32(feat_8)
        # feat_16_32 = self.down_16_32(feat_16)

        feat_32_fuse = self.sa_8_32(feat_8_32, feat_32)
        # feat_8_fuse = self.ca_32_8(feat_32_8, feat_8)

        # feat_8_fuse = self.relu(feat_8 + feat_16_8 + feat_32_8) # 1/8
        # feat_16_fuse = self.relu(feat_16 + feat_8_16 + feat_32_16) # 1/16
        # feat_32_fuse = self.relu(feat_32 + feat_8_32 + feat_16_32) # 1/32

        # feat_16_fuse_up = F.interpolate(feat_16_fuse, scale_factor=2.0, mode='nearest') # 1/8
        feat_32_fuse_up = F.interpolate(feat_32_fuse, scale_factor=4.0, mode='nearest') # 1/8
        
        feat_fuse = torch.cat([feat_8, feat_32_fuse_up], dim=1) # 1/8
        # feat_fuse = torch.cat([feat_8, feat_8_fuse], dim=1) # 1/8
        # feat_fuse = torch.cat([feat_8, feat_8_fuse, feat_32_fuse_up], dim=1) # 1/8
        
        feat_out = self.seg_head(feat_fuse)

        out = F.interpolate(feat_out, size=(
            H, W), mode='bilinear', align_corners=True)

        return out


if __name__ == "__main__":
    input = torch.rand(1, 3, 512, 1024).cuda()
    model = TriSeNet(classes=19).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('TriNet_v1:', output.size())
