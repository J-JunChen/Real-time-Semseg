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
        
        proj_value_trans = proj_value.permute(0, 2, 1)
        v_energy = torch.bmm(proj_value_trans, proj_value)
        value_attention = self.softmax(v_energy)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out, attention, value_attention
        # return out


class SANet(nn.Module):
    def __init__(self, layers=50, dropout=0.1, classes=2, zoom_factor=8):
        super(SANet, self).__init__()
        assert layers in [18, 34, 50, 101]
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor

        if layers == 18:
            resnet = models.resnet18(pretrained=True, deep_base=False, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
            # resnet = models.resnet18(pretrained=True, deep_base=False, strides=(1, 2, 1, 1), dilations=(1, 1, 2, 4))
        elif layers == 34:
            resnet = models.resnet34(pretrained=True, deep_base=False, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
        elif layers == 50:
            resnet = models.resnet50_semseg(pretrained=True, deep_base=True, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
            # resnet = models.resnet50_semseg(pretrained=True, deep_base=True, strides=(1, 2, 1, 1), dilations=(1, 1, 2, 4))
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

        self.sa_in_conv = ConvBNReLU(fea_dim, down_dim, ks=3, stride=1, padding=1)
        self.sa = SelfAttentionBlock(down_dim)
        self.sa_out_conv = ConvBNReLU(down_dim, down_dim, ks=3, stride=1, padding=1)
        self.sa_cls_seg = nn.Sequential(
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

        sa_feat = self.sa_in_conv(x)
        sa_feat, qk_attn, vv_attn = self.sa(sa_feat)
        # sa_feat = self.sa(sa_feat)
        sa_feat = self.sa_out_conv(sa_feat)
        sa_out = self.sa_cls_seg(sa_feat)

        if self.zoom_factor != 1:
            x = F.interpolate(sa_out, size=(h, w), mode='bilinear',
                              align_corners=True)
            # x_sa = F.interpolate(sa_out, size=(h, w), mode='bilinear',
            #                   align_corners=True)
            # x_cam = F.interpolate(cam_out, size=(h, w), mode='bilinear',
            #                   align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(
                    h, w), mode='bilinear', align_corners=True)
            return x, qk_attn, vv_attn ,aux
            # return x, aux
        else:
            return x, qk_attn, vv_attn
            # return x


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    input = torch.rand(4, 3, 225, 225).cuda()
    model = SANet(layers=50, dropout=0.1, classes=21, zoom_factor=8).cuda()
    model.eval()
    print(model)
    output,_ = model(input)
    print('SANet', output.size())
