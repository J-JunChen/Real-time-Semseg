# paper title: Pyramid scene parsing network (CVPR 2017)
# paper link: https://arxiv.org/abs/1612.01105
# reference code: https://github.com/hszhao/semseg

import torch
from torch import nn
import torch.nn.functional as F

import model.resnet as models


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True):
        super(PSPNet, self).__init__()
        assert layers in [18, 34, 50, 101]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm

        if layers == 18:
            resnet = models.resnet18(pretrained=True, deep_base=False, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
        elif layers == 34:
            resnet = models.resnet34(pretrained=True, deep_base=False, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
        elif layers == 50:
            # resnet = models.resnet50_semseg(pretrained=True, deep_base=True, strides=(1, 2, 1, 1), dilations=(1, 1, 2, 4))
            resnet = models.resnet50_semseg(pretrained=True, deep_base=True, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
        elif layers == 101:
            # resnet = models.resnet101_semseg(pretrained=True, deep_base=True, strides=(1, 2, 1, 1), dilations=(1, 1, 2, 4))
            resnet = models.resnet101_semseg(pretrained=True, deep_base=True, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))

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
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, down_dim, kernel_size=3, padding=1, bias=False),
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
        # x_size = x.size()
        # assert (x_size[2]-1) % 8 == 0 and (x_size[3] -
        #                                    1) % 8 == 0  # examine for H and W
        # h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        # w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
        h, w = x.size()[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        # if self.zoom_factor != 1:
        x = F.interpolate(x, size=(h, w), mode='bilinear',
                            align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            # if self.zoom_factor != 1:
            aux = F.interpolate(aux, size=(
                h, w), mode='bilinear', align_corners=True)
            return x, aux
        else:
            return x


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    input = torch.rand(4, 3, 225, 225).cuda()
    model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=8, use_ppm=True).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('PSPNet', output.size())
