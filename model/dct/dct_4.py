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


class DCTNet(nn.Module):
    def __init__(self, layers=50, dropout=0.1, classes=2,  vec_dim=300):
        super(DCTNet, self).__init__()
        assert layers in [18, 34, 50, 101]
        assert classes > 1
        self.vec_dim = vec_dim
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
            nn.Conv2d(fea_dim+3, down_dim, kernel_size=3, padding=1, bias=False),
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

    def dct_(self, x, h, w):
        out=F.interpolate(x, size=(h,w), mode='bilinear', align_corners=True)
        #print(out.shape)
        #dct_out_1 =torch.Tensor([cv2.dct(x[i,0,:,:].detach().cpu().numpy()) \
        #                          for i in range(x.shape[0])])
        dct_out_1 =torch.Tensor([cv2.dct(np.float32(out[i,0,:,:].detach().cpu().numpy())) \
                                    for i in range(x.shape[0])])
        dct_out_2 =torch.Tensor([cv2.dct(np.float32(out[i,1,:,:].detach().cpu().numpy())) \
                                    for i in range(x.shape[0])])
        dct_out_3 =torch.Tensor([cv2.dct(np.float32(out[i,2,:,:].detach().cpu().numpy())) \
                                    for i in range(x.shape[0])])
        dct_out=torch.zeros(size=(x.shape[0],3, h, w))
        dct_out[:,0,:,:]=dct_out_1 
        dct_out[:,1,:,:]=dct_out_2
        dct_out[:,2,:,:]=dct_out_3
        dct_out=dct_out.cuda()#放回cuda
        # out=dct_out.view(x.shape[0], 3, -1)
        # out=F.glu(out,dim=-1)
        # dct_out=out.view(x.shape[0], 1, -1)
        return dct_out
    
    def dct_torch(self, x, h, w):
        out=F.interpolate(x, size=(h,w), mode='bilinear', align_corners=True)
        out = out.to(torch.float32)
        
        return torch_dct.dct_2d(out, norm='ortho')

    def forward(self, x):
        N, C, H, W = x.size()
        feat = self.layer0(x)
        feat_4 = self.layer1(feat)
        feat_8 = self.layer2(feat_4) # spatial information
        feat_16 = self.layer3(feat_8)
        feat_32 = self.layer4(feat_16)
        H32, W32 = feat_32.size()[2:]

        dct_out=self.dct_(x, H32, W32)
        feat_fuse = torch.cat([feat_32, dct_out], dim=1)  
       
        logits = self.cls(feat_fuse)
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
