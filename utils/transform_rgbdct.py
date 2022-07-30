import random
import math
import numpy as np
import numbers
import collections
import cv2
from scipy import fftpack
# from scipy.fftpack.basic import fft

import torch

class Compose(object):
    # Composes segtransforms: segtransform.Compose([segtransform.RandScale([0.5, 2.0]), segtransform.ToTensor()])
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image, label):
        for t in self.segtransform:
            image, label = t(image, label)
        return image, label


class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, image_list, label):
        if not isinstance(image_list[0], np.ndarray) or not isinstance(image_list[1], np.ndarray) \
             or not isinstance(label, np.ndarray):
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                                "[eg: data readed by cv2.imread()].\n"))
        # if len(image.shape) > 3 or len(image.shape) < 2:
        #     raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims. \n"))
        # if len(image.shape) == 2:
        #     image = np.expand_dims(image, axis=2)
        # if not len(label.shape) == 2:
        #     raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray label with 2 dims.\n"))

        image_list[0] = torch.from_numpy(image_list[0].transpose((2, 0, 1)))
        image_list[1] = torch.from_numpy(image_list[1].transpose((2, 0, 1)))
        if not isinstance(image_list[0], torch.FloatTensor):
            image_list[0] = image_list[0].float()
        if not isinstance(image_list[1] , torch.FloatTensor):
            image_list[1] = image_list[1].float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        return image_list, label


class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean_rgb, mean_dct, std_rgb=None, std_dct=None):
        # if std is None:
        #     assert len(mean) > 0
        # else:
        #     assert len(mean) == len(std)
        self.mean_rgb = mean_rgb
        self.std_rgb = std_rgb
        self.mean_dct = mean_dct
        self.std_dct = std_dct

    def __call__(self, image_list, label):
        if self.std_rgb is None:
            for t, m in zip(image_list, self.mean_rgb):
                t.sub_(m)
        else:
            for t, m, s in zip(image_list[0], self.mean_rgb, self.std_rgb):
                t.sub_(m).div_(s)
            for t, m, s in zip(image_list[1], self.mean_dct, self.std_dct):
                t.sub_(m).div_(s)
        return image_list, label

    
class Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, image, label):
        image = cv2.resize(image, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.size[::-1], interpolation=cv2.INTER_NEAREST)
        return image, label


class RandScale(object):
    # Randomly resize image & label with scale factor in [scale_min, scale_max]
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2) 
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("segtransform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError('segtransform.RandScale() aspect_ratio param error.\n'))

    def __call__(self, image, label):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale * temp_aspect_ratio
        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        return image, label


class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image, label):
        h, w = label.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransform.Crop() need padding while padding argument is None\n"))
            image_rgb = cv2.copyMakeBorder(image[:,:,0:3], pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
            image_yuv = cv2.copyMakeBorder(image[:,:,3:6], pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
            image = np.concatenate((image_rgb, image_yuv), axis=2)
            label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)
        h, w = label.shape
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        image = image[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        label = label[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        return image, label


class RandRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError('segtransform.RandRotate() scale param error.\n'))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image_rgb = cv2.warpAffine(image[:,:,0:3], matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            image_yuv = cv2.warpAffine(image[:,:,3:6], matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            image = np.concatenate((image_rgb, image_yuv), axis=2)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
        return image, label


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label
    

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        return image, label


class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius
    
    def __call__(self, image, label):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label
    

class RGB2BGR(object):
    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label

class BGR2RGB(object):
    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label
    

class GetDctCoefficient(object):
    def __init__(self, 
                 block_size=8, 
                 sub_sampling='4:2:0', 
                 quality_factor=99, 
                 threshold=0.0):
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

    def __call__(self, image, label):
        # 1. Ensure the size of image and label satisfy the factor of 8.
        image = self.block_cropping(image, self.block_size)
        label = self.block_cropping(label, self.block_size)
        # 2. The chrominance channels Cr and Cb are subsampled
        if self.sub_sampling == '4:2:0':
            imSub = self.subsample_chrominance(image[:,:,3:6], 2, 2)
        # 3. Get the quatisation matrices, which will be applied to the DCT coefficients
        Q = self.quality_factorize(self.QY, self.QC, self.quality_factor)
        # 4. Apply DCT algorithm for orignal image
        TransAll, TransAllThresh ,TransAllQuant = self.dct_encoder(imSub, Q, self.block_size, self.thresh)
        # 5. Split the same frequency in each 8x8 blocks to the same channel
        dct_list = self.split_frequency(TransAll, self.block_size)
        # 6. upsample the Cr & Cb channel to concatenate with Y channel
        dct_coefficients = self.upsample(dct_list)
        image_list = [image[:,:,0:3], dct_coefficients]
        return image_list, label
    
    def block_cropping(self, image, blocksize=8):
        B = blocksize
        # print('orginal image shape:', image.shape)
        h, w = (np.array(image.shape[:2]) // B * B).astype(int)
        image = image[:h, :w]
        # print('modified image shape:', image.shape)
        return image
    
    def subsample_chrominance(self, YCbCr_image, SSV=2, SSH=2):
        crf = cv2.boxFilter(YCbCr_image[:,:,1], ddepth=-1, ksize=(2,2))
        cbf = cv2.boxFilter(YCbCr_image[:,:,2], ddepth=-1, ksize=(2,2))
        crsub = crf[::SSV, ::SSH] # sample with stride SSV in row and stride SSH in col.
        cbsub = cbf[::SSV, ::SSH]
        imSub_list = [YCbCr_image[:, :, 0], crsub, cbsub]
        return imSub_list

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
            vis0=vis0-128 # before DCT the pixel values of all channels are shifted by -128
            blocks = self.blockshaped(vis0, blocksize, blocksize)
            # dct_blocks = fftpack.dct(fftpack.dct(blocks, axis=1, norm='ortho'), axis=2, norm='ortho')
            dct_blocks = fftpack.dctn(blocks, axes=(-2,-1), norm='ortho')
            thres_blocks = dct_blocks * \
                (abs(dct_blocks) > thresh * np.amax(dct_blocks, axis=(1,2))[:, np.newaxis, np.newaxis]) # need to broadcast
            quant_blocks = np.round(thres_blocks / Q[idx])
        
            TransAll_list.append(self.unblockshaped(dct_blocks, channelrows, channelcols))
            TransAllThresh_list.append(self.unblockshaped(thres_blocks, channelrows, channelcols))
            TransAllQuant_list.append(self.unblockshaped(quant_blocks, channelrows, channelcols))
        return TransAll_list, TransAllThresh_list ,TransAllQuant_list
    
    def idct_decoder(self, TransAllQuant_list, Q, blocksize=8):
        h, w = TransAllQuant_list[0].shape
        c = len(TransAllQuant_list)
        B = blocksize
        DecAll=np.zeros((h, w, c), np.uint8)
        for idx,channel in enumerate(TransAllQuant_list):
            channelrows=channel.shape[0]
            channelcols=channel.shape[1]
            blocks = self.blockshaped(channel, B, B)
            dequantblocks = blocks * Q[idx]
            idct_blocks = fftpack.idctn(dequantblocks, axes= (-2,-1), norm='ortho')
            idct_blocks = np.round(idct_blocks)+128 # inverse shiftign of the shift of the pixel values, sucht that their value range is [0,...,255].
            idct_blocks[idct_blocks>255]=255
            idct_blocks[idct_blocks<0]=0
            idct_arr = self.unblockshaped(idct_blocks, channelrows, channelcols).astype(np.uint8)
            if idx != 0:
                idct_arr=cv2.resize(idct_arr,(w,h)) # the subsampled chrominance channels are interpolated, using cv2.INTER_LINEAR in default. 
            DecAll[:,:,idx]=np.round(idct_arr)
        return DecAll
    
    def label_idct(self, TransAll, blocksize=8):
        B = blocksize
        h, w = TransAll.shape
        blocks = self.blockshaped(TransAll, B, B)
        idct_blocks = fftpack.idctn(blocks, axes=(-2,-1), norm='ortho')
        idct_blocks = np.round(idct_blocks)+128 # inverse shiftign of the shift of the pixel values, sucht that their value range is [0,...,255].
        idct_blocks[idct_blocks>255]=255
        idct_blocks[idct_blocks<0]=0
        idct_arr = self.unblockshaped(idct_blocks, h, w).astype(np.uint8)
        return idct_arr
    
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
    
    def fuse_frequency(self, DctBlock_list, blocksize=8):
        Trans_list = []
        B = blocksize
        for idx, channel in enumerate(DctBlock_list):
            blocksV  = channel.shape[0]
            blocksH  = channel.shape[1]
            channelrows = blocksV * B
            channelcols = blocksH * B
            blocks = channel.reshape(blocksV * blocksH, B, B)
            idct_blocks = self.unblockshaped(blocks, channelrows, channelcols)
            Trans_list.append(idct_blocks)
        return Trans_list

    def upsample(self, DctBlock_list):
        h, w, c = DctBlock_list[0].shape
        DctUpAll = np.zeros((h, w, 3*c), np.float32)
        for idx, channel in enumerate(DctBlock_list):
            if idx == 0:
                DctUpAll[:, :, idx*c:(idx+1)*c] = channel
            else: 
                dct_block = cv2.resize(channel, (w,h))
                DctUpAll[:, :, idx*c:(idx+1)*c] = dct_block
        return DctUpAll


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
