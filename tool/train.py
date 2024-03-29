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
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from model.pspnet import PSPNet
from model.bisenet_v1 import BiseNet
from model.nonlocal_net import Nonlocal
from model.danet import DANet
from model.sanet import SANet

from utils import dataset, transform, config
from utils.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, find_free_port, save_checkpoint
from loss.kd_loss import KDLoss, KD_SA_Loss
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
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    # if args.arch == 'psp' or args.arch == 'nonlocal':
    #     assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    # elif args.arch == 'bise_v1':
    #     assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    # elif args.arch == 'psa':
    #     pass
    # else:
    #     raise Exception('architecture not supported yet'.format(args.arch))

def main():
    args = get_parser()
    check(args)
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
        import warnings
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    print(
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
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
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
        if args.arch == 'psp':
            teacher_model = PSPNet(layers=args.teacher_layers, classes=args.classes, zoom_factor=args.zoom_factor)
        elif args.arch == 'danet':
            teacher_model = DANet(layers=args.teacher_layers, classes=args.classes, zoom_factor=args.zoom_factor)
        elif args.arch == 'sanet':
            teacher_model = SANet(layers=args.teacher_layers, classes=args.classes, zoom_factor=args.zoom_factor)
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
    elif args.arch == 'danet':
        model = DANet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.pam_in_conv, model.pam, model.pam_out_conv, model.pam_cls_seg, 
                        model.cam_in_conv, model.cam, model.cam_out_conv, model.cam_cls_seg, 
                        model.cls_seg, model.aux]
    elif args.arch == 'sanet':
        model = SANet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.sa_in_conv, model.sa, model.sa_out_conv, model.sa_cls_seg, 
                       model.aux]
    elif args.arch == 'bise_v1':
        model = BiseNet(layers=args.layers, num_classes=args.classes, with_sp=args.with_sp)
        if args.with_sp:
            modules_ori = [model.sp, model.cp]
        else: 
            modules_ori = [model.cp]
        modules_new = [model.ffm, model.conv_out, model.conv_out16, model.conv_out32]
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
    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
    args.index_split = 5
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
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=True)
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
    
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(gpu)
    # criterion = OhemCELoss(thresh=0.7, ignore_index=args.ignore_label).cuda(gpu)

    kd_criterion = None
    if teacher_model is not None:
        if args.arch == 'sanet':
            # kd_criterion = []
            # kd_criterion.append(KDLoss(ignore_index=args.ignore_label).cuda(gpu))
            # kd_criterion.append(KD_SA_Loss().cuda(gpu))
            kd_criterion = KD_SA_Loss().cuda(gpu)
        else:
            kd_criterion = KDLoss(ignore_index=args.ignore_label).cuda(gpu)
        # kd_criterion = KDLoss(ignore_index=args.ignore_label).cuda(gpu)
            
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
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_mIoU_val = checkpoint['best_mIoU_val']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
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
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    if args.evaluate:
        val_transform = transform.Compose([
            transform.Crop([args.train_h, args.train_w], crop_type='center', padding=mean, ignore_label=args.ignore_label),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
        val_data = dataset.SemData(split='val', data_root=args.data_root, data_list=args.val_list, transform=val_transform)
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        if args.distributed:
            # Use .set_epoch() method to reshuffle the dataset partition at every iteration
            train_sampler.set_epoch(epoch)
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, teacher_model, criterion, kd_criterion, optimizer, epoch)
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)
        
        is_best = False
        if args.evaluate:
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)
            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)

                if best_mIoU_val < mIoU_val:
                    is_best = True
                    best_mIoU_val = mIoU_val
                    logger.info('==>The best val mIoU: %.3f' % (best_mIoU_val))

        
        if (epoch_log % args.save_freq == 0) and main_process():
            save_checkpoint(
                {
                    'epoch': epoch_log, 
                    'state_dict': model.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'best_mIoU_val': best_mIoU_val
                }, 
                is_best, 
                args.save_path
            )
            if is_best:
                logger.info('Saving checkpoint to:' + args.save_path + '/best.pth with mIoU: ' + str(best_mIoU_val) )
            else:
                logger.info('Saving checkpoint to:' + args.save_path + '/last.pth with mIoU: ' + str(mIoU_val) )

    if main_process():  
        writer.close() # it must close the writer, otherwise it will appear the EOFError!
        logger.info('==>Training done!\nBest mIoU: %.3f' % (best_mIoU_val))


def train(train_loader, model, teacher_model, criterion, kd_criterion, optimizer, epoch):
    # torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    # switch to train mode
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    end = time.time()
    max_iter = args.epochs * len(train_loader) # initialize for poly learning rate 
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.zoom_factor != 8:
            h = int((target.size()[1] - 1) / 8 * args.zoom_factor + 1)
            w = int((target.size()[2] - 1) / 8 * args.zoom_factor + 1) 
            # 'nearest' mode doesn't support align_corners mode and 'bilinear' mode is fine for downsampling
            target = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='bilinear', align_corners=True).squeeze(1).long()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if args.arch == 'bise_v1':
            main_out, aux_out1, aux_out2 = model(input) 
        elif args.arch == 'sanet':
            main_out, qk_attn, vv_attn, aux_out = model(input) 
        else:
            main_out, aux_out = model(input)
        # elif args.arch == 'danet':
        #     main_out, pam_out, cam_out, aux_out = model(input)
        if teacher_model is not None:
            with torch.no_grad():
                if args.arch == 'sanet':
                    teacher_out, tqk_attn, tvv_attn = teacher_model(input)
                else:
                    teacher_out = teacher_model(input)
            if args.arch == 'sanet':
                kd_loss = criterion(main_out, target)
                at_loss = kd_criterion(qk_attn, tqk_attn)
                vr_loss = kd_criterion(vv_attn, tvv_attn)
                # main_loss = kd_loss + at_loss + vr_loss
                main_loss = kd_loss + at_loss + vr_loss
                # main_loss = kd_criterion(main_out, target, teacher_out, qk_attn, vv_attn, tqk_attn, tvv_attn, alpha=args.alpha, temperature=args.temperature)
                del teacher_out, tqk_attn, tvv_attn
            else:
                main_loss = kd_criterion(main_out, target, teacher_out, alpha=args.alpha, temperature=args.temperature)
                del teacher_out # delete the teacher_out for releasing the gpu memory.
            # with torch.no_grad():
            #     teacher_out = teacher_model(input)
            # main_loss = kd_criterion(main_out, target, teacher_out, alpha=args.alpha, temperature=args.temperature)
            # del teacher_out # delete the teacher_out for releasing the gpu memory.
        else:
            main_loss = criterion(main_out, target)
            # if args.arch == 'danet':
            #     pam_loss = criterion(pam_out, target)
            #     cam_loss = criterion(cam_out, target)
            #     main_loss = main_loss + pam_loss + cam_loss
        if args.arch == 'bise_v1':
            aux_loss1 = criterion(aux_out1, target)
            aux_loss2 = criterion(aux_out2, target)
            aux_loss = aux_loss1 + aux_loss2
        else:
            aux_loss = criterion(aux_out, target)

        if not args.multiprocessing_distributed:
            main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)
        loss = main_loss + args.aux_weight * aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0)
        if args.multiprocessing_distributed:
            main_loss, aux_loss, loss = main_loss.detach() * n, aux_loss * n, loss * n  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(main_loss), dist.all_reduce(aux_loss), dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            main_loss, aux_loss, loss = main_loss / n, aux_loss / n, loss / n
        
        main_out = main_out.detach().max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(main_out, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
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

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss {aux_loss_meter.val:.4f} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                                                data_time=data_time,
                                                                                batch_time=batch_time,
                                                                                remain_time=remain_time,
                                                                                main_loss_meter=main_loss_meter,
                                                                                aux_loss_meter=aux_loss_meter,
                                                                                loss_meter=loss_meter,
                                                                                accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)  
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return main_loss_meter.avg, mIoU, mAcc, allAcc
        

def validate(val_loader, model, criterion):
    # torch.backends.cudnn.enabled = False  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    if main_process():
        logger.info('>>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            data_time.update(time.time() - end)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            if args.teacher_model_path != None and args.arch == 'sanet':
                output, _, _ = model(input)
            else:
                output = model(input)
            if args.zoom_factor != 8:
                output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
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
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % args.print_freq == 0) and main_process():
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                            data_time=data_time,
                                                            batch_time=batch_time,
                                                            loss_meter=loss_meter,
                                                            accuracy=accuracy))
    
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
