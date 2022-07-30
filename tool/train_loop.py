from ast import arg
from model.fftnet import FFTNet
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
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from model import *

from utils import dataset, transform, config
from utils.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, find_free_port, save_checkpoint
from loss.kd_loss import KDLoss
from loss.ohem_loss import OhemCELoss

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def get_parser():
    parser = argparse.ArgumentParser(description='Pytorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/cityscapes/cityscapes_transunet.yaml', help='config file')
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
    logger.handlers = [] # This is the key thing for the question!
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False # https://stackoverflow.com/questions/11820338/replace-default-handler-of-python-logger/11821510
    return logger


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    # https://github.com/Microsoft/human-pose-estimation.pytorch/issues/8
    # https://discuss.pytorch.org/t/training-performance-degrades-with-distributeddataparallel/47152/7
    # torch.backends.cudnn.enabled = False

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

        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed 
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        train(args.train_gpu, args.ngpus_per_node, args)


def train(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu 
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    teacher_model = None
    if args.teacher_model_path:
        teacher_model = PSPNet(layers=args.teacher_layers, classes=args.classes, zoom_factor=args.zoom_factor)
        kd_path = 'alpha_' + str(args.alpha) + '_Temp_' + str(args.temperature)
        args.save_path = os.path.join(args.save_path, kd_path)
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
    if args.arch == 'psp':
        model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.ppm, model.cls, model.aux]
    elif args.arch == 'nonlocal':
        model = Nonlocal(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.conv1, model.conv2, model.nl_block, model.cls, model.aux]
    elif args.arch == 'fanet':
        model = FANet(layers=args.layers, classes=args.classes)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.fa, model.fa_cls_seg, model.aux]
    elif args.arch == 'fftnet':
        model = FFTNet(layers=args.layers, classes=args.classes)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.freq, model.fa_cls_seg, model.aux]
    elif args.arch == 'fftnet_23':
        model = FFTNet23(layers=args.layers, classes=args.classes)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.freq, model.fa_cls_seg, model.aux]
    elif args.arch == 'dct':
        model = DCTNet(layers=args.layers, classes=args.classes, vec_dim=300)
        # modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3]
        # modules_new = [model.up_conv, model.cls, model.aux]  # DCT1
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        # modules_new = [model.ffm, model.cls, model.aux]   # DCT2
        # modules_new = [model.dct, model.cls, model.aux]  # DCT3
        modules_new = [model.cls, model.aux]  # DCT4
    elif args.arch == 'bise_v1':
        model = BiseNet(layers=args.layers, classes=args.classes, with_sp=args.with_sp)
        if args.with_sp:
            modules_ori = [model.sp, model.cp]
        else: 
            modules_ori = [model.cp]
        modules_new = [model.ffm, model.conv_out, model.conv_out16, model.conv_out32]
    elif args.arch == 'triple':
        model = TriSeNet(layers=args.layers, classes=args.classes)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        # modules_new = [model.down_8_32, model.down_16_32, model.sa_8_32, model.sa_16_32, model.seg_head]
    elif args.arch == 'triple_1':
        model = TriSeNet1(layers=args.layers, classes=args.classes)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        # modules_new = [model.arm16, model.arm32, model.gap, model.conv1x1_gap, model.conv_head32, model.conv_head16, model.ffm, model.seg_head]
    elif args.arch == 'ppm':
        model = PPM_Net(backbone=args.backbone, layers=args.layers, classes=args.classes)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
    elif args.arch == 'fc':
        model = FC_Net(backbone=args.backbone, layers=args.layers, classes=args.classes)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
    elif args.arch == 'transunet':
        from model.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
        from model.vit_seg_modeling import TransUnet

        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.classes
        config_vit.n_skip = args.n_skip
        args.img_size = args.train_h
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        model = TransUnet(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        model.load_from(weights=np.load(config_vit.pretrained_path))
        modules_ori = [model.transformer]
        modules_new = [model.decoder, model.segmentation_head]
    # modules_new = []
    # for key, value in model._modules.items():
    #     if "layer" not in key:
    #         modules_new.append(value)
    args.index_split = len(modules_ori)  # the module after index_split need multiply 10 at learning rate 
    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
    optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if teacher_model is not None:
            teacher_model = nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path) # tensorboardX
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
        if teacher_model is not None:
            logger.info(teacher_model)
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=False)
        if teacher_model is not None:
            teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model.cuda(), device_ids=[gpu])

    else:
        model = torch.nn.DataParallel(model.cuda())
        if teacher_model is not None:
            teacher_model = torch.nn.DataParallel(teacher_model.cuda())
    
    if teacher_model is not None:
        checkpoint = torch.load(args.teacher_model_path, map_location=lambda storage, loc: storage.cuda())
        teacher_model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loading teacher checkpoint '{}'".format(args.teacher_model_path))
    
    if args.use_ohem:
        criterion = OhemCELoss(thresh=0.7, ignore_index=args.ignore_label).cuda(gpu)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(gpu)

    kd_criterion = None
    if teacher_model is not None:
        kd_criterion = KDLoss(ignore_index=args.ignore_label).cuda(gpu)
            
    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight: '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> mp weight found at '{}'".format(args.weight))
    
    best_mIoU_val = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            # Load all tensors onto GPU
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_iter = checkpoint['iteration']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_mIoU_val = checkpoint['best_mIoU_val']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (iteration {})".format(args.resume, checkpoint['iteration']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))    
        
    value_scale = 255
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

    train_data = dataset.SemData(split='train', data_root=args.data_root, data_list=args.train_list, transform=train_transform)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, \
        shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, \
        sampler=train_sampler, drop_last=True)
    if args.evaluate:
        # val_h = int(args.base_h * args.scale)
        # val_w = int(args.base_w * args.scale)
        val_transform = transform.Compose([
            transform.Crop([args.train_h, args.train_w], crop_type='center', padding=mean, ignore_label=args.ignore_label),
            # transform.Resize(size=(val_h, val_w)),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
        val_data = dataset.SemData(split='val', data_root=args.data_root, data_list=args.val_list, transform=val_transform)
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, \
            shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    # Training Loop
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    # switch to train mode
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
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
            if args.distributed:
                train_sampler.set_epoch(epoch)
                # if main_process():
                #     logger.info('train_sampler.set_epoch({})'.format(epoch))
            data_iter = iter(train_loader)
            input, target = next(data_iter)
            # need to update the AverageMeter for new epoch
            main_loss_meter = AverageMeter()
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
       
        if not args.multiprocessing_distributed:
            main_loss = torch.mean(main_loss)
        loss = main_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0)
        if args.multiprocessing_distributed:
            main_loss,  loss = main_loss.detach() * n, loss * n  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(main_loss), dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            main_loss, loss = main_loss / n, loss / n
        
        main_out = main_out.detach().max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(main_out, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
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
        if iter_log % args.print_freq == 0 and main_process():
            logger.info('Iter [{}/{}] '
                        'LR: {lr:.3e}, '
                        'ETA: {remain_time}, '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f}), '
                        'Batch: {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                        'MainLoss: {main_loss_meter.val:.4f}, '
                        'Loss: {loss_meter.val:.4f}, '
                        'Accuracy: {accuracy:.4f}.'.format(iter_log, args.max_iter,
                                                                lr=current_lr,
                                                                remain_time=remain_time,
                                                                data_time=data_time,
                                                                batch_time=batch_time,
                                                                main_loss_meter=main_loss_meter,
                                                                loss_meter=loss_meter,
                                                                accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', main_loss_meter.val, iter_log)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), iter_log)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), iter_log)
            writer.add_scalar('allAcc_train_batch', accuracy, iter_log)
        
        if iter_log % len(train_loader) == 0 or iter_log == max_iter: # for each epoch or the max interation
            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)  
            accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            mIoU_train = np.mean(iou_class)
            mAcc_train = np.mean(accuracy_class)
            allAcc_train = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
            loss_train = main_loss_meter.avg
            if main_process():
                logger.info('Train result at iteration [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'\
                    .format(iter_log, max_iter, mIoU_train, mAcc_train, allAcc_train))
                writer.add_scalar('loss_train', loss_train, iter_log)
                writer.add_scalar('mIoU_train', mIoU_train, iter_log)
                writer.add_scalar('mAcc_train', mAcc_train, iter_log)
                writer.add_scalar('allAcc_train', allAcc_train, iter_log)

        # if iter_log % args.save_freq == 0:
            is_best = False
            if args.evaluate:
                loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)
                model.train() # the mode change from eval() to train()
                if main_process():
                    writer.add_scalar('loss_val', loss_val, iter_log)
                    writer.add_scalar('mIoU_val', mIoU_val, iter_log)
                    writer.add_scalar('mAcc_val', mAcc_val, iter_log)
                    writer.add_scalar('allAcc_val', allAcc_val, iter_log)

                    if best_mIoU_val < mIoU_val:
                        is_best = True
                        best_mIoU_val = mIoU_val
                        logger.info('==>The best val mIoU: %.3f' % (best_mIoU_val))

            if main_process():
                save_checkpoint({
                        'iteration': iter_log, 
                        'state_dict': model.state_dict(), 
                        'optimizer': optimizer.state_dict(),
                        'best_mIoU_val': best_mIoU_val
                    }, is_best, args.save_path)
                logger.info('Saving checkpoint to:{}/last.pth with mIoU:{:.3f}'\
                    .format(args.save_path, mIoU_val))
                if is_best:
                    logger.info('Saving checkpoint to:{}/best.pth with mIoU:{:.3f}'\
                        .format(args.save_path, best_mIoU_val))

    if main_process():  
        writer.close() # it must close the writer, otherwise it will appear the EOFError!
        logger.info('==>Training done! The best val mIoU during training: %.3f' % (best_mIoU_val))


def validate(val_loader, model, criterion):
    # torch.backends.cudnn.enabled = False  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    if main_process():
        logger.info('>>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>>')
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    # end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # data_time.update(time.time() - end)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(input)
            loss = criterion(output, target)

            n = input.size(0)
            if args.multiprocessing_distributed:
                loss = loss * n # not considering ignore pixels
                count = target.new_tensor([n], dtype=torch.long)
                dist.all_reduce(loss), dist.all_reduce(count)
                n = count.item()
                loss = loss / n
            else:
                loss = torch.mean(loss)
            
            output = output.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), input.size(0))
            # batch_time.update(time.time() - end)
            # end = time.time()
            # if ((i + 1) % args.print_freq == 0) and main_process():
            #     logger.info('Test: [{}/{}] '
            #                 'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
            #                 'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
            #                 'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
            #                 'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
            #                                                 data_time=data_time,
            #                                                 batch_time=batch_time,
            #                                                 loss_meter=loss_meter,
            #                                                 accuracy=accuracy))
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc

if __name__ == "__main__":
    main()
