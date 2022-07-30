
# paper title: Pyramid scene parsing network (CVPR 2017)
# paper link: https://arxiv.org/abs/1612.01105
# reference code: https://github.com/hszhao/semseg

from model.densenet import densenet169
import torch
from torch import nn
import torch.nn.functional as F

# from model.resnet import resnet18, resnet34, resnet50_semseg, resnet101_semseg
# from model.densenet import densenet121, densenet161
from torchvision.models import resnet50, resnet101, resnet152
from torchvision.models import densenet121, densenet161


class FC_Net(nn.Module):
    def __init__(self, backbone='densenet', layers=50, dropout=0.1, classes=2):
        super(FC_Net, self).__init__()
        assert classes > 1
        if backbone == 'resnet':
            if layers == 50:
                resnet = resnet50(pretrained=True)
            elif layers == 101:
                resnet = resnet101(pretrained=True)
            elif layers == 152:
                resnet = resnet152(pretrained=True)

            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
            
            if layers == 18 or layers == 34:
                fea_dim = 512
            else:
                fea_dim = 2048
        elif backbone == 'densenet':
            if layers == 121:
                densenet = densenet121(pretrained=True)
                fea_dim = 1024
            elif layers == 161:
                densenet = densenet161(pretrained=True)
                fea_dim = 2208
            self.layer0 = nn.Sequential(densenet.features.conv0, densenet.features.norm0, \
                densenet.features.relu0, densenet.features.pool0)
            self.layer1 = densenet.features.denseblock1
            self.layer2 = nn.Sequential(densenet.features.transition1, densenet.features.denseblock2)
            self.layer3 = nn.Sequential(densenet.features.transition2, densenet.features.denseblock3)
            self.layer4 = nn.Sequential(densenet.features.transition3, densenet.features.denseblock4)
        down_dim = fea_dim // 4
        
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, down_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(down_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(down_dim, classes, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.size()[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
  
        x = self.cls(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear',
                            align_corners=True)
        
        return x


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    input = torch.rand(4, 3, 225, 225).cuda()
    model = FC_Net(backbone='densenet', layers=161, dropout=0.1, classes=21).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('PPM_Net', output.size())
