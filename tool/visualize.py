import os
import random
import time
import cv2
import numpy as np
import logging
import argparse
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset

import cityscapesscripts.helpers.labels as CSLabels
import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as CSEval
from PIL import Image

from model import *

from utils import transform, config
from utils.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, save_checkpoint
from loss.kd_loss import KDLoss
from loss.ohem_loss import OhemCELoss
from loss.detail_loss import DetailAggregateLoss

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='Pytorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/cityscapes/visualize.yaml', help='config file')
    parser.add_argument('opts', help='see config/cityscapes/cityscapes_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError('Image list file do not exist:' + data_list + '\n'))
    image_label_list = []
    # list_read = open(data_list).readlines()
    f = open(data_list)
    list_read = []
    for i in range(16):
        list_read.append(f.readline())
    f.close()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("".join(list_read))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list

class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) # BGR 3 channel ndarray with shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label

def main():
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True

    if args.manual_seed is not None:  # manual_seed for random
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    args.ngpus_per_node = len(args.train_gpu)
    train(args.train_gpu, args.ngpus_per_node, args)


def train(gpu, ngpus_per_node, argss):
    global args
    args = argss

    if args.arch == 'triple':
        model = TriSeNet(layers=args.layers, classes=args.classes)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        # modules_new = [model.down_8_32, model.sa_8_32, model.seg_head]
        modules_new = []
        for key, value in model._modules.items():
            if "layer" not in key:
                modules_new.append(value)
    args.index_split = len(modules_ori)  # the module after index_split need multiply 10 at learning rate 
    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
    optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    print("=> creating model ...")
    print("Classes: {}".format(args.classes))
    print(model)
    model = torch.nn.DataParallel(model.cuda())
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(gpu)

    value_scale = 255
    ## RGB mean & std
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.ignore_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    train_data = SemData(split='train', data_root=args.data_root, data_list=args.train_list, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, \
        shuffle=True, num_workers=args.workers, pin_memory=True, \
        sampler=None, drop_last=True)
    
    # Training Loop
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.max_iter

    data_iter = iter(train_loader)
    epoch = 0
    for current_iter in range(args.start_iter, args.max_iter):
        try:
            input, target = next(data_iter)
            if not input.size(0) == args.batch_size:
                raise StopIteration
        except StopIteration:
            epoch += 1
            data_iter = iter(train_loader)
            input, target = next(data_iter)
            # need to update the AverageMeter for new epoch
            main_loss_meter = AverageMeter()
            aux_loss_meter = AverageMeter()
            loss_meter = AverageMeter()
            intersection_meter = AverageMeter()
            union_meter = AverageMeter()
            target_meter = AverageMeter()
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        main_out = model(input)
        main_loss = criterion(main_out, target)
        aux_loss = torch.tensor(0).cuda()
        loss = main_loss + args.aux_weight * aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n = input.size(0)
        main_out = main_out.detach().max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(main_out, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # Using Poly strategy to change the learning rate
        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        for index in range(0, args.index_split): # args.index_split = 5 -> ResNet has 5 stages
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10
        
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        iter_log = current_iter + 1

        if iter_log % args.print_freq == 0:
            print('Iteration: [{}/{}] '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'ETA {remain_time} '
                    'MainLoss {main_loss_meter.val:.4f} '
                    'AuxLoss {aux_loss_meter.val:.4f} '
                    'Loss {loss_meter.val:.4f} '
                    'Accuracy {accuracy:.4f}.'.format(iter_log, args.max_iter,
                                                            data_time=data_time,
                                                            batch_time=batch_time,
                                                            remain_time=remain_time,
                                                            main_loss_meter=main_loss_meter,
                                                            aux_loss_meter=aux_loss_meter,
                                                            loss_meter=loss_meter,
                                                            accuracy=accuracy))
    save_checkpoint({
                'iteration': iter_log, 
                'state_dict': model.state_dict(), 
                'optimizer': optimizer.state_dict(),
            }, False, args.save_path)

def evaluate():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    if args.arch == 'triple':
        model = TriSeNet(layers=args.layers, classes=args.classes)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    if os.path.isfile(args.model_path):
        print("=> loading checkpoint '{}'".format(args.model_path))
        # checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    
    value_scale = 255
    ## RGB mean & std
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_mean = [item * value_scale for item in rgb_mean]
    rgb_std = [0.229, 0.224, 0.225]
    rgb_std = [item * value_scale for item in rgb_std]

    val_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=rgb_mean, std=rgb_std)])
    train_data = SemData(split='train', data_root=args.data_root, data_list=args.train_list, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(train_data, batch_size=1, \
        shuffle=False, num_workers=args.workers, pin_memory=True)
    
    print('>>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>>')

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
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(input)
            _, H, W = target.shape
            output = output.detach().max(1)[1]
            results.append(output.cpu().numpy().reshape(H,W))
            intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)
            end = time.time()
            print('Val: [{}/{}] '
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

    print('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(args.classes):
        print('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    print('<<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<<<<<')
    print('Convert to Label ID')
    result_files = results2img(results=results, data_root=args.data_root, data_list=args.train_list, save_dir='./visualization/train_result', to_label_id=True)
    print('Convert to Label ID Finished')


def _convert_to_label_id(result):
    """Convert trainId to id for cityscapes."""
    if isinstance(result, str):
        result = np.load(result)
    result_copy = result.copy()
    for trainId, label in CSLabels.trainId2label.items():
        result_copy[result == trainId] = label.id

    return result_copy

def results2img(results, data_list, data_root, save_dir, to_label_id):
    """Write the segmentation results to images.

    Args:
        results (list[list | tuple | ndarray]): Testing results of the
            dataset.
        imgfile_prefix (str): The filename prefix of the png files.
            If the prefix is "somepath/xxx",
            the png files will be named "somepath/xxx.png".
        to_label_id (bool): whether convert output to label_id for
            submission

    Returns:
        list[str: str]: result txt files which contains corresponding
        semantic segmentation images.
    """
    assert isinstance(results, list), 'results must be a list'
    list_read = []
    f = open(data_list)
    for i in range(16):
        list_read.append(f.readline())
    # list_read = open(data_list).readlines()
    assert len(list_read) == len(results)
    filenames = []
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        
        image_name = os.path.join(data_root, line_split[0])
        filenames.append(image_name)
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    result_files = []
    for idx in range(len(results)):
        result = results[idx]
        if to_label_id:
            result = _convert_to_label_id(result)
        filename = filenames[idx]
        basename = os.path.splitext(os.path.basename(filename))[0]

        png_filename = os.path.join(save_dir, f'{basename}.png')

        output = Image.fromarray(result.astype(np.uint8)).convert('P')
        palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
        for label_id, label in CSLabels.id2label.items():
            palette[label_id] = label.color

        output.putpalette(palette) # if you want to see te color result
        output.save(png_filename)
        result_files.append(png_filename)

    return result_files

if __name__ == '__main__':
    main()
    evaluate()