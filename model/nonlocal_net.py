# paper title: Non-Local Neural Networks
# paper link: https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.html

import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import NonLocal2d

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


class Nonlocal(nn.Module):
    def __init__(self, layers=50, dropout=0.1, classes=2, zoom_factor=8, reduction=2, use_scale=True, mode='embedded_gaussian'):
        super(Nonlocal, self).__init__()
        assert layers in [18, 34, 50, 101]
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.reduction = reduction
        self.use_scale = use_scale
        self.mode = mode

        if layers == 18:
            resnet = models.resnet18(pretrained=True, deep_base=False, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
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

        self.conv1 = ConvBNReLU(fea_dim, down_dim, ks=3, stride=1, padding=1)
        self.nl_block = NonLocal2d(
            in_channels=down_dim,
            reduction=self.reduction,
            use_scale=self.use_scale,
            norm_cfg = dict(type='SyncBN', requires_grad=True),
            mode=self.mode)
        self.conv2 = ConvBNReLU(down_dim, down_dim, ks=3, stride=1, padding=1)
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim + down_dim, down_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(down_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(down_dim, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(aux_dim, aux_dim // 4, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(aux_dim // 4),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(aux_dim // 4, classes, kernel_size=1)
            )

    def forward(self, x):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3] -
                                           1) % 8 == 0  # examine for H and W
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x_feat = self.layer4(x_tmp)
        x = self.conv1(x_feat)
        x = self.nl_block(x)
        x = self.conv2(x)
        x = self.cls(torch.cat([x_feat, x], dim=1))
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear',
                              align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(
                    h, w), mode='bilinear', align_corners=True)
            return x, aux
        else:
            return x

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    input = torch.rand(4, 3, 225, 225).cuda()
    model = Nonlocal(layers=50, dropout=0.1, classes=21, zoom_factor=8).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('Nonlocal Net: ', output.size())