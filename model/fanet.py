import torch
from torch import nn
import torch.nn.functional as F

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


class FastAttentionBlock(nn.Module):
    """ following .utils.SelfAttentionBlock

    Args:
        in_channels (int): Input channels of key, query, value feature.
        channels (int): Output channels of key/query transform.
        smf_channels (int): Output channels.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_num_convs (int): Number of convs for value projection.
    """
    def __init__(self, in_channels, channels, up_channels,
                 smf_channels, up_flag, smf_flag):
        super(FastAttentionBlock, self).__init__()
        self.up_flag = up_flag
        self.smf_flag = smf_flag
        
        self.key_project = ConvBNReLU(
            in_channels, 
            channels, 
            ks=1,
            stride=1, 
            padding=0,
            with_relu=False)    
        self.query_project = ConvBNReLU(
            in_channels, 
            channels, 
            ks=1,
            stride=1, 
            padding=0,
            with_relu=False)   
        self.value_project = ConvBNReLU(
            in_channels, 
            in_channels, 
            ks=1, 
            stride=1, 
            padding=0)
        self.out_project = ConvBNReLU(
            in_channels,
            in_channels,
            ks=1,
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

    def forward(self, x, up_feat_in):
        """Forward function."""
        batch_size, channels, height, width = x.size()

        query = self.query_project(x)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()
        # query_ = query.view(*query.shape[:2], -1).permute(0, 2, 1)
        query = F.normalize(query, p=2, dim=2, eps=1e-12) # l2 norm for query along the channel dimension

        key = self.key_project(x)
        key = key.reshape(*key.shape[:2], -1)
        # key_ = key.view(*key.shape[:2], -1)
        key = F.normalize(key, p=2, dim=1, eps=1e-12) # l2 norm for key along the channel dimension

        value = self.value_project(x)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()
        # value = value.view(*value.shape[:2], -1).permute(0, 2, 1)

        sim_map = torch.matmul(key, value) 
        # sim_map /= (height * width) # cosine similarity
        context = torch.matmul(query, sim_map)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *x.shape[2:])
        # context = context.view(batch_size, -1, *x.shape[2:])

        context = self.out_project(context)
        fuse_feature = context + x

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


class FANet(nn.Module):
    def __init__(self, layers=50, dropout=0.1, classes=2):
        super(FANet, self).__init__()
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
            
        self.fa = nn.ModuleList()
        for i in range(6,10): # the number of in_channels is 2^i
            self.fa.append(
                FastAttentionBlock(
                    in_channels=2 ** i,
                    channels=32,  # channels can be changed if you want.
                    up_channels=2 ** i if i == 6 else 2 ** (i-1),
                    # smf_channels=2 ** i if i is 6 else 2 ** (i-1), 
                    smf_channels=128, 
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

        up_feat_32 = self.fa[3](feat_32, None)
        up_feat_16, smf_feat_16 = self.fa[2](feat_16, up_feat_32)
        up_feat_8 = self.fa[1](feat_8, up_feat_16)
        smf_feat_4 = self.fa[0](feat_4, up_feat_8)

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
    input = torch.rand(4, 3, 225, 225).cuda()
    model = FANet(layers=18, dropout=0.1, classes=21).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('FANet', output.size())
