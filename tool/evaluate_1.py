#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import argparse

import os
import os.path as osp
import logging
import time
import numpy as np
import math
from tqdm import tqdm

from model import *
from utils import dataset, transform, config

def get_parser():
    parser = argparse.ArgumentParser(description='Pytorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/cityscapes/cityscapes_fanet.yaml', help='config file')
    parser.add_argument('opts', help='see config/cityscapes/cityscapes_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = 'main-logger'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

class MscEval(object):
    def __init__(self,
            model,
            dataloader,
            scales = [ 1.0],
            n_classes = 19,
            lb_ignore = 255,
            cropsize = 1024,
            flip = False,
            *args, **kwargs):
        self.scales = scales
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.flip = flip
        self.cropsize = cropsize
        ## dataloader
        self.dl = dataloader
        self.net = model


    def pad_tensor(self, inten, size):
        N, C, H, W = inten.size()
        outten = torch.zeros(N, C, size[0], size[1]).cuda()
        outten.requires_grad = False
        margin_h, margin_w = size[0]-H, size[1]-W
        hst, hed = margin_h//2, margin_h//2+H
        wst, wed = margin_w//2, margin_w//2+W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]


    def eval_chip(self, crop):
        with torch.no_grad():
            out = self.net(crop)[0]
            prob = F.softmax(out, 1)
            if self.flip:
                crop = torch.flip(crop, dims=(3,))
                out = self.net(crop)[0]
                out = torch.flip(out, dims=(3,))
                prob += F.softmax(out, 1)
            #prob = torch.exp(prob)
        return prob


    def crop_eval(self, im):
        cropsize = self.cropsize
        stride_rate = 1.0
        N, C, H, W = im.size()
        long_size, short_size = (H,W) if H>W else (W,H)
        if long_size < cropsize:
            im, indices = self.pad_tensor(im, (cropsize, cropsize))
            prob = self.eval_chip(im)
            prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
        else:
            stride = math.ceil(cropsize*stride_rate)
            if short_size < cropsize:
                if H < W:
                    im, indices = self.pad_tensor(im, (cropsize, W))
                else:
                    im, indices = self.pad_tensor(im, (H, cropsize))
            N, C, H, W = im.size()
            n_x = math.ceil((W-cropsize)/stride)+1
            n_y = math.ceil((H-cropsize)/stride)+1
            prob = torch.zeros(N, self.n_classes, H, W).cuda()
            prob.requires_grad = False
            for iy in range(n_y):
                for ix in range(n_x):
                    hed, wed = min(H, stride*iy+cropsize), min(W, stride*ix+cropsize)
                    hst, wst = hed-cropsize, wed-cropsize
                    chip = im[:, :, hst:hed, wst:wed]
                    prob_chip = self.eval_chip(chip)
                    prob[:, :, hst:hed, wst:wed] += prob_chip
            if short_size < cropsize:
                prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
        return prob


    def scale_crop_eval(self, im, scale):
        N, C, H, W = im.size()
        new_hw = [int(H*scale), int(W*scale)]
        im = F.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(im)
        prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
        return prob


    def compute_hist(self, pred, lb):
        n_classes = self.n_classes
        ignore_idx = self.lb_ignore
        keep = np.logical_not(lb==ignore_idx)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes**2)
        hist = hist.reshape((n_classes, n_classes))
        return hist


    def evaluate(self):
        ## evaluate
        n_classes = self.n_classes
        hist = np.zeros((n_classes, n_classes), dtype=np.float32)
        dloader = tqdm(self.dl)
        if dist.is_initialized() and not dist.get_rank()==0:
            dloader = self.dl
        for i, (imgs, label) in enumerate(dloader):
            N, H, W = label.shape
            label = label.reshape(N, 1, H, W)
            print(label.shape)
            probs = torch.zeros((N, self.n_classes, H, W))
            probs.requires_grad = False
            imgs = imgs.cuda()
            for sc in self.scales:
                prob = self.scale_crop_eval(imgs, sc)
                probs += prob.detach().cpu()
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)

            hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))
            # hist_once = self.compute_hist(preds, label.data.numpy())
            hist = hist + hist_once
        IOUs = np.diag(hist) / (np.sum(hist, axis=0)+np.sum(hist, axis=1)-np.diag(hist))
        mIOU = np.mean(IOUs)
        return mIOU


def evaluate(respth='./res', dspth='/data2/.encoding/data/cityscapes', checkpoint=None):
    args = get_parser()
    ## logger
    logger = get_logger()

    ## model
    logger.info('\n')
    logger.info('===='*20)
    logger.info('evaluating the model ...\n')
    logger.info('setup and restore model')
    n_classes = 19
    net = FANet(layers=18, classes=n_classes)

    # if checkpoint is None:
    #     save_pth = osp.join(respth, 'model_final.pth')
    # else:
    #     save_pth = checkpoint

    # net.load_state_dict(torch.load(save_pth))

    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        # checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
        checkpoint = torch.load(args.model_path)
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    net.cuda()
    net.eval()

    ## dataset
    # batchsize = 1
    # n_workers = 2
    # dsval = CityScapes(dspth, mode='val')
    # dl = DataLoader(dsval,
    #                 batch_size = batchsize,
    #                 shuffle = False,
    #                 num_workers = n_workers,
    #                 drop_last = False)

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]   
    val_transform = transform.Compose([
        # transform.Crop([args.test_h, args.test_w], crop_type='center', padding=mean, ignore_label=args.ignore_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    val_data = dataset.SemData(split='val', data_root=args.data_root, data_list=args.val_list, transform=val_transform)
    dl = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    ## evaluator
    logger.info('compute the mIOU')
    evaluator = MscEval(net, dl, scales=[1.0],flip=False)
    ## eval
    mIOU = evaluator.evaluate()
    logger.info('mIOU is: {:.6f}'.format(mIOU))



if __name__ == "__main__":
    evaluate()