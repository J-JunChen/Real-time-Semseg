# paper title: Dual Attention Network for Scene Segmentation (CVPR 2019)
# paper link:https://arxiv.org/pdf/1809.02983
# reference code: https://github.com/junfu1115/DANet

import torch
from torch import nn
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

class PAM(nn.Module):
    """ Position attention module"""
    def __init__(self, in_dim):
        super(PAM, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class DANet(nn.Module):
    def __init__(self, layers=50, dropout=0.1, classes=2, zoom_factor=8):
        super(DANet, self).__init__()
        assert layers in [18, 34, 50, 101]
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor

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

        self.pam_in_conv = ConvBNReLU(fea_dim, down_dim, ks=3, stride=1, padding=1)
        self.pam = PAM(down_dim)
        self.pam_out_conv = ConvBNReLU(down_dim, down_dim, ks=3, stride=1, padding=1)
        self.pam_cls_seg = nn.Sequential(
            nn.Dropout2d(p=dropout),
            nn.Conv2d(down_dim, classes, kernel_size=1)
        )

        self.cam_in_conv = ConvBNReLU(fea_dim, down_dim, ks=3, stride=1, padding=1)
        self.cam = CAM(down_dim)
        self.cam_out_conv = ConvBNReLU(down_dim, down_dim, ks=3, stride=1, padding=1)
        self.cam_cls_seg = nn.Sequential(
            nn.Dropout2d(p=dropout),
            nn.Conv2d(down_dim, classes, kernel_size=1)
        )

        self.cls_seg = nn.Sequential(
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
        x = self.layer4(x_tmp)

        pam_feat = self.pam_in_conv(x)
        pam_feat = self.pam(pam_feat)
        pam_feat = self.pam_out_conv(pam_feat)
        pam_out = self.pam_cls_seg(pam_feat)

        cam_feat = self.cam_in_conv(x)
        cam_feat = self.cam(cam_feat)
        cam_feat = self.cam_out_conv(cam_feat)
        cam_out = self.cam_cls_seg(cam_feat)

        feat_sum = pam_feat + cam_feat
        x = self.cls_seg(feat_sum)

        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear',
                              align_corners=True)
            # x_pam = F.interpolate(pam_out, size=(h, w), mode='bilinear',
            #                   align_corners=True)
            # x_cam = F.interpolate(cam_out, size=(h, w), mode='bilinear',
            #                   align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(
                    h, w), mode='bilinear', align_corners=True)
            # return x, x_pam, x_cam, aux
            return x, aux
        else:
            return x


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    input = torch.rand(4, 3, 225, 225).cuda()
    model = DANet(layers=50, dropout=0.1, classes=21, zoom_factor=8).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('DANet', output.size())
