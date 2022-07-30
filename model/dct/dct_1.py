# paper title: Pyramid scene parsing network (CVPR 2017)
# paper link: https://arxiv.org/abs/1612.01105
# reference code: https://github.com/hszhao/semseg
import numpy as np
import torch
from torch import nn
import torch.fft as fft
import torch.nn.functional as F
import torch_dct

import model.resnet as models
# from dct_encoding import DctMaskEncoding

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


class DCTModule(nn.Module):
    def __init__(self, vec_dim, mask_size=128):
        super(DCTModule, self).__init__()
        self.vec_dim = vec_dim
        # self.dct_vector_coords = self.get_dct_vector_coords(r=mask_size)

    def get_dct_vector_coords(self, r=128):
        """
        Get the coordinates with zigzag order.
        """
        dct_index = []
        for i in range(r):
            if i % 2 == 0:  # start with even number
                index = [(i-j, j) for j in range(i+1)]
                dct_index.extend(index)
            else:
                index = [(j, i-j) for j in range(i+1)]
                dct_index.extend(index)
        for i in range(r, 2*r-1):
            if i % 2 == 0:
                index = [(i-j, j) for j in range(i-r+1,r)]
                dct_index.extend(index)
            else:
                index = [(j, i-j) for j in range(i-r+1,r)]
                dct_index.extend(index)
        dct_idxs = np.asarray(dct_index)
        return dct_idxs

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        dct_vector_coords_all = self.get_dct_vector_coords(r=height)
        dct_vector_coords = dct_vector_coords_all[:self.vec_dim]
        # masks = x.to(dtype=float)
        masks = x.to(torch.float32)
        dct_all = torch_dct.dct_2d(masks, norm='ortho')
        xs, ys = dct_vector_coords[:, 0], dct_vector_coords[:, 1]
        dct_vectors = dct_all[:,:, xs, ys]  # reshape as vector
        return dct_vectors  # [batch_size, channels, D]


class DCTNet(nn.Module):
    def __init__(self, layers=50, dropout=0.1, classes=2, use_dct=True, vec_dim=300):
        super(DCTNet, self).__init__()
        assert layers in [18, 34, 50, 101]
        assert classes > 1
        self.use_dct = use_dct
        self.vec_dim = vec_dim

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
        self.layer1, self.layer2, self.layer3 = resnet.layer1, resnet.layer2, resnet.layer3
        
        if layers == 18 or layers == 34:
            fea_dim = 512
            aux_dim = 256
        else:
            fea_dim = 2048
            aux_dim = 1024
        down_dim = fea_dim // 4
        if use_dct:
            self.dct_encoding = DCTModule(vec_dim=self.vec_dim)
            self.up_conv = ConvBNReLU(self.vec_dim, fea_dim, ks=1, stride=1, padding=0)
            
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
        N, C, H, W = x.size()
        feat = self.layer0(x)
        feat_4 = self.layer1(feat)
        feat_8 = self.layer2(feat_4) # spatial information
        feat_16 = self.layer3(feat_8)
        # feat_32 = self.layer4(feat_16)
        if self.use_dct:
            feat_dct = self.dct_encoding(feat_16) # feature -> DCT vector
            h = int(np.sqrt(feat_dct.size()[1]))
            feat_dct = feat_dct.permute(0, 2, 1).contiguous()
            feat_dct_matrice = feat_dct.view(N, self.vec_dim, h, -1)  # DCT vector -> DCT matrice
            feat_dct_up = self.up_conv(feat_dct_matrice)

        logits = self.cls(feat_dct_up)
        out = F.interpolate(logits, size=(H, W), mode='bilinear',
                            align_corners=True)

        if self.training:
            aux = self.aux(feat_16)
            aux = F.interpolate(aux, size=(
                H, W), mode='bilinear', align_corners=True)
            return out, aux
        else:
            return out


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    input = torch.rand(4, 3, 512, 512).cuda()
    model = DCTNet(layers=18,  dropout=0.1, classes=21, vec_dim=100).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('DCTNet', output.size())
