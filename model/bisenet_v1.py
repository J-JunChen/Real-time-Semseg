# paper title: BisNet: Bilateral segmentation network for real-time semantic segmentation
# paper link: https://openaccess.thecvf.com/content_ECCV_2018/html/Changqian_Yu_BisNet_Bilateral_Segmentation_ECCV_2018_paper.html

import torch
from torch import nn
import torch.nn.functional as F

from model.resnet import resnet18, resnet50, resnet101


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


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionRefinementModule, self).__init__()
        self.conv3x3_bn_relu = ConvBNReLU(
            in_channels, out_channels, ks=3, stride=1, padding=1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        feature = self.conv3x3_bn_relu(x)
        attention = self.channel_attention(feature)
        out = torch.mul(feature, attention)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1):
        super(FeatureFusionModule, self).__init__()
        self.conv1x1_bn_relu = ConvBNReLU(
            in_channels, out_channels, ks=1, stride=1, padding=0)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        feature = self.conv1x1_bn_relu(x)
        attention = self.channel_attention(feature)
        y = torch.mul(feature, attention)
        out = feature + y
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
    def __init__(self, in_channels, out_channels, backbone=resnet18):
        super(ContextPath, self).__init__()
        inner_channels = 128
        # follow the original paper of ResNet, when deep_base is False.
        self.backbone = backbone(pretrained=True, deep_base=False)
        self.arm16 = AttentionRefinementModule(256, inner_channels)
        self.arm32 = AttentionRefinementModule(512, inner_channels)

        # the initial layer conv is 7x7, instead of three 3x3
        self.init_layer = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool
        )
        # stage channels for resnet18 and resnet50 are:(64, 128, 256, 512)
        self.layer1, self.layer2, self.layer3, self.layer4 \
            = self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.conv1x1_gap = nn.Conv2d(
            512, inner_channels, kernel_size=1, stride=1, padding=0)

        self.conv_head32 = ConvBNReLU(
            inner_channels, out_channels, ks=3, stride=1, padding=1)  # didn't find in the paper
        self.conv_head16 = ConvBNReLU(
            inner_channels, out_channels, ks=3, stride=1, padding=1)

    def forward(self, x):
        x = self.init_layer(x)
        x = self.layer1(x)
        feat_8 = self.layer2(x)  # 1/8 of spatial size
        feat_16 = self.layer3(feat_8)  # 1/16 of spatial size
        feat_32 = self.layer4(feat_16)  # 1/32 of spatial size
        H8, W8 = feat_8.size()[2:]
        H16, W16 = feat_16.size()[2:]
        H32, W32 = feat_32.size()[2:]

        gap_feat = self.gap(feat_32)
        conv_gap_feat = self.conv1x1_gap(gap_feat)
        up_gap_feat = F.interpolate(conv_gap_feat, size=(
            H32, W32), mode='bilinear', align_corners=True)

        feat_32_arm = self.arm32(feat_32)
        feat_32_sum = feat_32_arm + up_gap_feat
        feat32_up = F.interpolate(feat_32_sum, size=(
            H16, W16), mode='bilinear', align_corners=True)
        feat32_up = self.conv_head32(feat32_up)

        feat_16_arm = self.arm16(feat_16)
        feat_16_sum = feat_16_arm + feat32_up
        feat16_up = F.interpolate(feat_16_sum, size=(
            H8, W8), mode='bilinear', align_corners=True)
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up  # x8, x16


class BiseNetHead(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super(BiseNetHead, self).__init__()
        self.conv3x3_bn_relu = ConvBNReLU(in_channels=in_channels, out_channels=mid_channels,
                                          ks=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(
            in_channels=mid_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        x = self.conv3x3_bn_relu(x)
        out = self.conv1x1(x)
        return out


class BiseNet(nn.Module):
    def __init__(self, num_classes):
        super(BiseNet, self).__init__()
        self.criterion = criterion
        
        self.sp = SpatialPath(in_channels=3, out_channels=128)
        self.cp = ContextPath(in_channels=3, out_channels=128)
        self.ffm = FeatureFusionModule(
            in_channels=256, out_channels=256)  # concat: 128+128
        self.conv_out = BiseNetHead(in_channels=256, mid_channels=256, num_classes=num_classes)

        if self.training:
            self.conv_out16 = BiseNetHead(in_channels=128, mid_channels=64, num_classes=num_classes)
            self.conv_out32 = BiseNetHead(in_channels=128, mid_channels=64, num_classes=num_classes)

    def forward(self, x, y=None):
        H, W = x.size()[2:]
        feat_sp = self.sp(x)
        feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)

        out = F.interpolate(feat_out, size=(
            H, W), mode='bilinear', align_corners=True)

        if self.training:
            feat_out16 = self.conv_out16(feat_cp8)
            feat_out32 = self.conv_out32(feat_cp16)

            out16 = F.interpolate(feat_out16, size=(
                H, W), mode='bilinear', align_corners=True)
            out32 = F.interpolate(feat_out32, size=(
                H, W), mode='bilinear', align_corners=True)

            return out, out16, out32
        else:
            return out


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIABLE_DEVICES'] = '0, 1'
    input = torch.rand(2, 3, 769, 769).cuda()
    model = BiseNet(num_classes=19).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('BiseNet_v1:', output.size())
