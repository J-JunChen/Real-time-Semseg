import os
import random
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data

from model import *

from utils import dataset, transform, config, dct_mean_std
from utils.util import AverageMeter, intersectionAndUnionGPU

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def get_parser():
    parser = argparse.ArgumentParser(description='Pytorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/cityscapes/cityscapes_trisenet.yaml', help='config file')
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


def evaluate():
    args = get_parser()
    logger = get_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    if args.arch == 'psp':
        model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor)
    elif args.arch == 'nonlocal':
        model = Nonlocal(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor)
    elif args.arch == 'danet':
        model = DANet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor)
    elif args.arch == 'sanet':
        model = SANet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor)
    elif args.arch == 'fanet':
        model = FANet(layers=args.layers, classes=args.classes)
    elif args.arch == 'fftnet':
        model = FFTNet(layers=args.layers, classes=args.classes)
    elif args.arch == 'fftnet_23':
        model = FFTNet23(layers=args.layers, classes=args.classes)
    elif args.arch == 'bise_v1':
        model = BiseNet(layers=args.layers, classes=args.classes, with_sp=args.with_sp)
    elif args.arch == 'dct':
        model = DCTNet(layers=args.layers, classes=args.classes, vec_dim=300)
    elif args.arch == 'triple':
        model = TriSeNet(layers=args.layers, classes=args.classes)
    elif args.arch == 'triple_1':
        model = TriSeNet1(layers=args.layers, classes=args.classes)
    elif args.arch == 'ppm':
        model = PPM_Net(backbone=args.backbone, layers=args.layers, classes=args.classes)
    elif args.arch == 'fc':
        model = FC_Net(backbone=args.backbone, layers=args.layers, classes=args.classes)

    logger.info(model)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        # checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
        
    value_scale = 255
    ## RGB mean & std
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_mean = [item * value_scale for item in rgb_mean]
    rgb_std = [0.229, 0.224, 0.225]
    rgb_std = [item * value_scale for item in rgb_std]

    # DCT mean & std
    dct_mean = dct_mean_std.train_upscaled_static_mean
    dct_mean = [item * value_scale for item in dct_mean]
    dct_std = dct_mean_std.train_upscaled_static_std
    dct_std = [item * value_scale for item in dct_std]

    val_h = int(args.base_h * args.scale)
    val_w = int(args.base_w * args.scale)

    val_transform = transform.Compose([
        transform.Resize(size=(val_h, val_w)),
        # transform.GetDctCoefficient(),
        transform.ToTensor(),
        transform.Normalize(mean=rgb_mean, std=rgb_std)])
    val_data = dataset.SemData(split='val', img_type='rgb', data_root=args.data_root, data_list=args.val_list, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # val_transform = transform.Compose([
    #     # transform.Resize(size=(val_h, val_w)),
    #     transform.GetDctCoefficient(),
    #     transform.ToTensor(),
    #     transform.Normalize(mean_rgb=rgb_mean, mean_dct=dct_mean, std_rgb=rgb_std, std_dct=dct_std)])
    # val_data = dataset.SemData(split='val', img_type='rgb&dct', data_root=args.data_root, data_list=args.val_list, transform=val_transform)
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # test_transform = transform.Compose([
    #     transform.ToTensor(),
    #     transform.Normalize(mean=mean, std=std)])
    # # test_transform = transform.Compose([transform.ToTensor()])
    # test_data = dataset.SemData(split='test', data_root=args.data_root, data_list=args.test_list, transform=test_transform)
    # val_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    logger.info('>>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    results = []

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            data_time.update(time.time() - end)
            # _, _, H, W = input.shape
            input = input.cuda(non_blocking=True)
            # input = [input[0].cuda(non_blocking=True), input[1].cuda(non_blocking=True)]
            target = target.cuda(non_blocking=True)
            # if args.scale != 1.0:
                # input = F.interpolate(input, size=(val_h, val_w), mode='bilinear', align_corners=True)
            if args.teacher_model_path != None and args.arch == 'sanet':
                output, _, _ = model(input)
            else:
                output = model(input)
                # if args.scale != 1.0:
                    # output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=True)
            _, H, W = target.shape
            # output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=True)
            output = output.detach().max(1)[1]
            results.append(output.cpu().numpy().reshape(H,W))
            intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % 10 == 0) or (i + 1 == len(val_loader)):
                logger.info('Val: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                            data_time=data_time,
                                                            batch_time=batch_time,
                                                            accuracy=accuracy))
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<<<<<')
    
    # logger.info('Convert to Label ID')
    # result_files = dataset.results2img(results=results, data_root=args.data_root, data_list=args.val_list, save_dir='./val_result', to_label_id=True)
    # logger.info('Convert to Label ID Finished')


def test():
    args = get_parser()
    logger = get_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]  

    test_h = int(args.base_h * args.scale)
    test_w = int(args.base_w * args.scale)

    test_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    # test_transform = transform.Compose([transform.ToTensor()])
    test_data = dataset.SemData(split='test', data_root=args.data_root, data_list=args.test_list, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    if args.arch == 'psp':
        model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor)
    elif args.arch == 'nonlocal':
        model = Nonlocal(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor)
    elif args.arch == 'danet':
        model = DANet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor)
    elif args.arch == 'sanet':
        model = SANet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor)
    elif args.arch == 'fanet':
        model = FANet(layers=args.layers, classes=args.classes)
    elif args.arch == 'bise_v1':
        model = BiseNet(layers=args.layers, classes=args.classes, with_sp=args.with_sp)
    elif args.arch == 'dct':
        model = DCTNet(layers=args.layers, classes=args.classes, use_dct=True, vec_dim=300)
    logger.info(model)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        # checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
        
    logger.info('>>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()

    model.eval()
    end = time.time()
    results = []

    with torch.no_grad():
        for i, (input, _) in enumerate(test_loader):
            data_time.update(time.time() - end)
            input = np.squeeze(input.numpy(), axis=0)
            image = np.transpose(input, (1, 2, 0))
            h, w, _ = image.shape
            prediction = np.zeros((h, w, args.classes), dtype=float)

            input = torch.from_numpy(image.transpose((2, 0, 1))).float()
            # for t, m, s in zip(input, mean, std):
            #     t.sub_(m).div_(s)
            input = input.unsqueeze(0).cuda()
            # if args.scale != 1.0:
            #     input = F.interpolate(input, size=(new_H, new_W), mode='bilinear', align_corners=True)
            if args.teacher_model_path != None and args.arch == 'sanet':
                output, _, _ = model(input)
            else:
                output = model(input)
                # if args.scale != 1.0:
                #     output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=True)
            output = F.softmax(output, dim=1)
            output = output[0]
            output = output.data.cpu().numpy()
            prediction = output.transpose(1, 2, 0)
            prediction = np.argmax(prediction, axis=2)
            results.append(prediction)
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)):
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
                                                                                        data_time=data_time,
                                                                                        batch_time=batch_time))
    
    logger.info('<<<<<<<<<<<<<<<<<< End Testing <<<<<<<<<<<<<<<<<<<<<')
    
    logger.info('Convert to Label ID')
    result_files = dataset.results2img(results=results, data_root=args.data_root, data_list=args.test_list, save_dir='./val_result', to_label_id=True)
    logger.info('Convert to Label ID Finished')

if __name__ == "__main__":
    evaluate()
    # test()
