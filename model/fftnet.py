import torch
from torch import nn
import torch.fft as fft
import torch.nn.functional as F

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

class FeatureFrequencySeparationModule(nn.Module):
    """Feature Frequency Separation Module
    """
    def __init__(self,
                 in_channels,
                 up_channels,
                 smf_channels,
                 high_ratio,
                 low_ratio,
                 up_flag, 
                 smf_flag):
        super(FeatureFrequencySeparationModule, self).__init__()
        self.in_channels = in_channels
        self.up_flag = up_flag
        self.smf_flag = smf_flag
        high_channels = int(self.in_channels * high_ratio)
        # low_channels = int(self.in_channels * low_ratio)
        # mid_channels = in_channels - high_channels - low_channels
        low_channels = self.in_channels - high_channels
        self.high_project = ConvBNReLU(
            self.in_channels,
            high_channels,
            1,
            stride=1,
            padding=0)
        self.low_project = ConvBNReLU(
            self.in_channels,
            low_channels,
            1,
            stride=1,
            padding=0)
        self.out_project = ConvBNReLU(
            2 * self.in_channels,
            self.in_channels,
            1,
            stride=1,
            padding=0)
        if self.up_flag:
            self.up = ConvBNReLU(
                in_channels,
                up_channels,
                1,
                stride=1,
                padding=1)
        if self.smf_flag:
            self.smooth = ConvBNReLU(
                in_channels,
                smf_channels,
                3,
                stride=1,
                padding=1)

    def upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        '''
        _, _, H, W = y.size()
        x = F.interpolate(
            x,
            size=(H, W),
            mode='bilinear',
            align_corners=True)
        return x + y
  
    def guassian_low_high_pass_filter(self, x, D0=10):
        """reference code: https://blog.csdn.net/weixin_43959755/article/details/115528425 
        """
        _, _, H, W = x.size()
        y, z = torch.meshgrid(torch.arange(H), torch.arange(W))
        center = ((H-1)//2, (W-1)//2)
        dis_square = (y - center[0])**2 + (z - center[1])**2
        low_filter = torch.exp((-1) * dis_square / (2 * (D0 ** 2))).cuda()
        high_filter = 1 - low_filter
        return x * low_filter, x * high_filter
          
    def forward(self, x, up_feat_in):
        # separate feature for two frequency
        freq_x = fft.fftn(x)
        freq_shift = fft.fftshift(freq_x)
        
        # low_freq_shift = self.easy_low_pass_filter(freq_x)
        # high_freq_shift = self.easy_high_pass_filter(freq_x)
        low_freq_shift, high_freq_shift = self.guassian_low_high_pass_filter(freq_shift)

        low_freq_ishift = fft.ifftshift(low_freq_shift)
        high_freq_ishift = fft.ifftshift(high_freq_shift)
        
        _low_freq_x = torch.abs(fft.ifftn(low_freq_ishift))
        _high_freq_x = torch.abs(fft.ifftn(high_freq_ishift))

        low_freq_x = self.low_project(_low_freq_x)
        high_freq_x = self.high_project(_high_freq_x)

        feat = torch.cat([x, low_freq_x, high_freq_x], dim=1)
        context =  self.out_project(feat)
        fuse_feature = context + x # Whether use skip connection or not
        
        if self.up_flag and self.smf_flag:
            if up_feat_in is not None:
                fuse_feature = self.upsample_add(up_feat_in, fuse_feature)
            up_feature = self.up(fuse_feature)
            smooth_feature = self.smooth(fuse_feature)
            return up_feature, smooth_feature
        
        if self.up_flag and not self.smf_flag:
            if up_feat_in is not None:
                fuse_feature = self.upsample_add(up_feat_in, fuse_feature)
            up_feature = self.up(fuse_feature)
            return up_feature
        
        if not self.up_flag and self.smf_flag:
            if up_feat_in is not None:
                fuse_feature = self.upsample_add(up_feat_in, fuse_feature)
            smooth_feature = self.smooth(fuse_feature)
            return smooth_feature
        

class FFTNet(nn.Module):
    def __init__(self, layers=18, dropout=0.1, classes=2):
        super(FFTNet, self).__init__()
        if layers == 18:
            resnet = models.resnet18(pretrained=True, deep_base=False, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
        elif layers == 34:
            resnet = models.resnet34(pretrained=True, deep_base=False, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
        elif layers == 50:
            resnet = models.resnet50_semseg(pretrained=True, deep_base=True, strides=(1, 2, 1, 1), dilations=(1, 1, 2, 4))

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
            
        self.freq = nn.ModuleList()
        for i in range(6,10): # the number of in_channels is 2^i
            self.freq.append(
                FeatureFrequencySeparationModule(
                    in_channels=2 ** i,
                    up_channels=2 ** i if i == 6 else 2 ** (i-1),
                    smf_channels=128,
                    high_ratio=1 - 0.2 * (i-5),
                    # high_ratio=0.5,
                    low_ratio=0.2,
                    up_flag=False if i == 6 else True ,
                    smf_flag=True if i%2 == 0 else False ,
                )
            )
        self.fa_cls_seg = nn.Sequential(
            nn.Dropout2d(p=dropout),
            nn.Conv2d(256, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(aux_dim, aux_dim // 4, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(aux_dim // 4),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(aux_dim // 4, classes, kernel_size=1)
            )

    def upsample_cat(self, x, y):
        '''Upsample and concatenate feature maps.
        '''
        _, _, H, W = y.size()
        x = F.interpolate(
            x,
            size=(H, W),
            mode='bilinear',
            align_corners=True
        )
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        _,_,h,w = x.size()

        x = self.layer0(x)
        feat_4 = self.layer1(x)
        feat_8 = self.layer2(feat_4)
        feat_16 = self.layer3(feat_8)
        feat_32 = self.layer4(feat_16)

        up_feat_32 = self.freq[3](feat_32, None)
        up_feat_16, smf_feat_16 = self.freq[2](feat_16, up_feat_32)
        up_feat_8 = self.freq[1](feat_8, up_feat_16)
        smf_feat_4 = self.freq[0](feat_4, up_feat_8)
        up_out = self.upsample_cat(smf_feat_16, smf_feat_4)
        fa_out = self.fa_cls_seg(up_out)
        output = F.interpolate(fa_out, size=(h, w), mode='bilinear',
                            align_corners=True)
        if self.training:
            aux = self.aux(feat_16)
            aux = F.interpolate(aux, size=(
                h, w), mode='bilinear', align_corners=True)
            return output, aux
        else:
            return output


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    input = torch.rand(4, 3, 224, 224).cuda()
    model = FFTNet(layers=18, dropout=0.1, classes=21).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('FFTNet', output.size())