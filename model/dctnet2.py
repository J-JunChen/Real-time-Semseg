# paper title: Pyramid scene parsing network (CVPR 2017)
# paper link: https://arxiv.org/abs/1612.01105
# reference code: https://github.com/hszhao/semseg
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch_dct
from scipy import fftpack
import cv2

# from model.resnet_dct import ResNetDCT_2345, ResNetDCT_345
from model.resnet import resnet18

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
    def __init__(self, 
                 layers=50, 
                 dropout=0.1, 
                 classes=2,
                 block_size=8, 
                 sub_sampling='4:2:0', 
                 quality_factor=99, 
                 threshold=0.0,
                 vec_dim=300):
        super(DCTNet, self).__init__()
        assert layers in [18, 34, 50, 101]
        assert classes > 1
        self.classes = classes
        self.block_size = block_size
        self.sub_sampling = sub_sampling
        self.quality_factor = quality_factor
        self.thresh = threshold
        # the quantisation matrices for the luminace channel (QY)
        self.QY=np.array([[16,11,10,16,24,40,51,61],
                                [12,12,14,19,26,48,60,55],
                                [14,13,16,24,40,57,69,56],
                                [14,17,22,29,51,87,80,62],
                                [18,22,37,56,68,109,103,77],
                                [24,35,55,64,81,104,113,92],
                                [49,64,78,87,103,121,120,101],
                                [72,92,95,98,112,100,103,99]])
        # the quantisation matrices for the chrominance channels (QC)
        self.QC=np.array([[17,18,24,47,99,99,99,99],
                                [18,21,26,66,99,99,99,99],
                                [24,26,56,99,99,99,99,99],
                                [47,66,99,99,99,99,99,99],
                                [99,99,99,99,99,99,99,99],
                                [99,99,99,99,99,99,99,99],
                                [99,99,99,99,99,99,99,99],
                                [99,99,99,99,99,99,99,99]])
        # Backbone
        # if layers in [18,34]:
        resnet = resnet18(pretrained=False, deep_base=False, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = \
             resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        
        if layers == 18 or layers == 34:
            fea_dim = 512
        else:
            fea_dim = 2048
        down_dim = fea_dim // 4

        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, down_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(down_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(down_dim, classes, kernel_size=1)
        )
        

    def forward(self, x):
        N, C, H, W = x.size()
        dct_x = self.dct_calculation(x)
        feat = self.layer0(dct_x)
        feat_4 = self.layer1(feat)
        feat_8 = self.layer2(feat_4) 
        feat_16 = self.layer3(feat_8)
        feat_32 = self.layer4(feat_16)
        H32, W32 = feat_32.size()[2:]

        logits = self.cls(feat_32)
        out = F.interpolate(logits, size=(H, W), mode='bilinear',
                            align_corners=True)
        return out
        # B = 8
        # out_ = out.permute(0, 2, 3, 1)
        # fuse_out = torch.zeros((N, self.classes, H*B, W*B)).cuda()
        # for i in range(self.classes):
        #     block = out_[:, :, :, i * 64 : (i+1) * 64]
        #     block1 = block.reshape(N, -1, B, B)
        #     block2 = block1.reshape(N, H, W, B, B)
        #     block3 = block2.permute(0,1,3,2,4)
        #     block4 = block3.reshape(N, H*B, W*B)

        #     fuse_out[:,i,:,:] = block4
        # return fuse_out
    
    def dct_calculation(self, x):
        N, C, H, W = x.size()
        B = self.block_size
        # # 2. The chrominance channels Cr and Cb are subsampled
        # # if self.sub_sampling == '4:2:0':
        # #     imSub = self.subsample_chrominance(x, 2, 2)
        # # 3. Get the quatisation matrices, which will be applied to the DCT coefficients
        # Q = self.quality_factorize(self.QY, self.QC, self.quality_factor)
        # # 4. Apply DCT algorithm for orignal image
        # TransAll, TransAllThresh ,TransAllQuant = self.dct_encoder(x, Q, self.block_size, self.thresh)
        # # 5. Split the same frequency in each 8x8 blocks to the same channel
        # dct_list = self.split_frequency(TransAll, self.block_size)
        # # 6. upsample the Cr & Cb channel to concatenate with Y channel
        # # dct_coefficients = self.upsample(dct_list)
        
        blocksV = int(W / B)
        blocksH = int(H / B)
        # vis0 = torch.zeros_like(x).cuda()
        # for row in range(blocksV):
        #     for col in range(blocksH):
        #         currentblock = torch_dct.dct_2d(x[:,:, row*B:(row+1)*B, col*B:(col+1)*B], norm='ortho')
        #         vis0[:,:, row*B:(row+1)*B, col*B:(col+1)*B] = currentblock
        
        block = x.reshape(N, C, blocksH, B, blocksV, B)
        block1 = block.permute(0,1,2,4,3,5)
        block2 = block1.reshape(N, C, -1, B, B)
        dct_block = torch_dct.dct_2d(block2, norm='ortho')
        block3 = dct_block.reshape(N, C, blocksH, blocksV, B, B)
        block4 = block3.permute(0,1,2,4,3,5)
        block5 = block4.reshape(N, C, H, W)

        # print(torch.allclose(vis0, block5))

        return block5
    
    def quality_factorize(self, qy, qc, QF=99):
        if QF < 50 and QF > 1:
            scale = np.floor(5000/QF)
        elif QF < 100:
            scale = 200-2*QF
        else:
            print("Quality Factor must be in the range [1..99]")
        scale = scale / 100.0
        # print("Q factor:{}, Q scale:{} ".format(QF, scale))
        q = [qy*scale, qc*scale, qc*scale]
        return q

    def dct_encoder(self, imSub_list, Q, blocksize=8, thresh=0.05):
        TransAll_list=[]
        TransAllThresh_list=[]
        TransAllQuant_list=[]
        for idx,channel in enumerate(imSub_list):
            channelrows=channel.shape[0]
            channelcols=channel.shape[1]
            vis0 = np.zeros((channelrows,channelcols), np.float32)
            vis0[:channelrows, :channelcols] = channel
            # vis0=vis0-128 # before DCT the pixel values of all channels are shifted by -128
            blocks = self.blockshaped(vis0, blocksize, blocksize)
            # dct_blocks = fftpack.dct(fftpack.dct(blocks, axis=1, norm='ortho'), axis=2, norm='ortho')
            dct_blocks = torch_dct.dct_2d(blocks, norm='ortho')
            thres_blocks = dct_blocks * \
                (abs(dct_blocks) > thresh * np.amax(dct_blocks, axis=(1,2))[:, np.newaxis, np.newaxis]) # need to broadcast
            quant_blocks = np.round(thres_blocks / Q[idx])
        
            TransAll_list.append(self.unblockshaped(dct_blocks, channelrows, channelcols))
            TransAllThresh_list.append(self.unblockshaped(thres_blocks, channelrows, channelcols))
            TransAllQuant_list.append(self.unblockshaped(quant_blocks, channelrows, channelcols))
        return TransAll_list, TransAllThresh_list ,TransAllQuant_list
    
    def split_frequency(self, Trans_list, blocksize=8):
        DctBlock_list = []
        B = blocksize
        for idx, channel in enumerate(Trans_list):
            channelrows = channel.shape[0]
            channelcols = channel.shape[1]
            blocksV = int(channelrows / B)
            blocksH = int(channelcols / B)  
            dct_blocks = self.blockshaped(channel, B, B)
            DctBlock_list.append(dct_blocks.reshape(blocksV, blocksH, B*B))        
        return DctBlock_list

    def blockshaped(self, arr, nrows, ncols):
        """
        Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size

        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
        assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                .swapaxes(1,2)
                .reshape(-1, nrows, ncols))

    def unblockshaped(self, arr, h, w):
        """
        Return an array of shape (h, w) where
        h * w = arr.size

        If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
        then the returned array preserves the "physical" layout of the sublocks.
        """
        n, nrows, ncols = arr.shape
        return (arr.reshape(h//nrows, -1, nrows, ncols)
                .swapaxes(1,2)
                .reshape(h, w))


if __name__ == "__main__":
    
    input = torch.rand(4, 3, 512, 512).cuda()
    # input = torch.rand(4, 192, 16, 16).cuda()
    # input = torch.rand(4, 192, 64, 64).cuda()
    model = DCTNet(layers=18, dropout=0.1, classes=5, vec_dim=300).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('DCTNet', output.size())
