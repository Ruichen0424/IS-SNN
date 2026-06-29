#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import time
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Subset
import torchvision.transforms as transforms
from timm.optim.optim_factory import create_optimizer
from spikingjelly.activation_based import neuron, functional
from torch.utils.tensorboard import SummaryWriter

from Network.ResNet.SEW_Resnet_BF import *
from utils import *


# In[ ]:


# Set Parser
parser = argparse.ArgumentParser("TrainNet")
################################# Basic settings #########################################################################
parser.add_argument('-bz', '--batch_size', type=int, default=256, help='batch size')
parser.add_argument('-dp', '--dataset_path', type=str, default='/ssd/Datasets/ImageNet/', help='dataset path')
parser.add_argument('--dataset', type=str, default='ImageNet', help='dataset')
parser.add_argument('-sp', '--save_path', type=str, default='./run', help='dataset path')
parser.add_argument('-lp', '--log_path', type=str, default='/log', help='log path')
parser.add_argument('-eo', '--exist_ok', action='store_false', help='exist_ok')
parser.add_argument('-n', '--name', type=str, default='sew34_bf_0.5_t4', help='experiment name')
parser.add_argument('--suffix', type=str, default='_bf', help='suffix')
parser.add_argument('-tk', '--topk', type=int, default=5, help='top_k')
parser.add_argument('-s', '--seed', type=int, default=2024, help='seed for initializing training')
parser.add_argument('-cn', '--class_number', type=int, default=1000, help='number of classes')
parser.add_argument('--deterministic', action='store_true', help='ensure experimental reproducibility')
parser.add_argument('--dummy', action='store_true', help='dummy')
parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')
################################# GPU settings #########################################################################
parser.add_argument('--world_size', type=int, default=1, help='number of nodes for distributed training')
parser.add_argument('--rank', type=int, default=0, help='node rank for distributed training')
parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:8080', help='url used to set up distributed training')
parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend')
parser.add_argument('--gpu', type=int, default=None, help='GPU id to use.')
parser.add_argument('--multiprocessing_distributed', action='store_true', help='Use multi-processing distributed training to launch'
                                                                                'N processes per node, which has N GPUs. This is the'
                                                                                'fastest way to use PyTorch for either single node or'
                                                                                'multi node data parallel training')
################################# Training settings #####################################################################
parser.add_argument('-e', '--epochs', type=int, default=128, help='num of training epochs')
parser.add_argument('-se', '--start_epoch', type=int, default=0, help='begin epoch')
parser.add_argument('--lr', type=float, default=0.2, help='init learning rate')
parser.add_argument('-m', '--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('-wd', '--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--opt', type=str, default='sgd', help='optimizer')
parser.add_argument('--amp', action='store_true', help='use amp')
parser.add_argument('--timestep', type=int, default=4, help='timestep')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha')
parser.add_argument('--mixup', action='store_true', help='use mixup')
parser.add_argument('--t_train', type=int, default=0, help='random temporal delete')
################################# Schedulers settings #####################################################################
parser.add_argument('--scheduler', type=str, default='cosineLR', help='scheduler, support [cosineLR, stepLR]')
parser.add_argument('--lr_min', type=float, default=0, help='minimum learning rate')
parser.add_argument('--warmup_t', type=int, default=0, help='warmup t')
parser.add_argument('--warmup_lr_init', type=float, default=0, help='warmup lr initial')
################################# other settings #########################################################
parser.add_argument('-w', '--workers', type=int, default=16, help='number of data loading workers (default: 4)')
parser.add_argument('-p', '--print_freq', type=int, default=200, help='print frequency (default: 20)')
parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')

args = parser.parse_args()

# In[ ]:


# Set Files
os.makedirs(args.save_path, exist_ok=True)                  # run path
file_path = args.save_path + '/' + args.name + '/'
os.makedirs(file_path, exist_ok=args.exist_ok)              # exp name path
weights_path = file_path + 'weights/'
os.makedirs(weights_path, exist_ok=args.exist_ok)           # weights path
log_file = file_path + args.log_path + '/'
os.makedirs(log_file, exist_ok=True)                        # log path

# Set Logger
logger = logging.getLogger(args.name)
setup_default_logging(log_path=log_file+'log.txt')
writer = SummaryWriter(log_dir=file_path)


# In[ ]:


def main():
    # Set Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger.info('Set Seed: {}'.format(args.seed))
    logger.info(f'args: {args}\n')
    # Set torch
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info('Using deterministic mode, which can slow down your training considerably!')
    else:
        torch.backends.cudnn.benchmark = True

    if args.gpu is not None:
        logger.info('You have chosen a specific GPU. This will completely disable data parallelism.')
        if not torch.cuda.is_available():
            logger.warning('Error: CUDA not found! Will use CPU for training.')
        
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
            
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            logger.warning("nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


# In[ ]:


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        
    ###################################### Create model #######################################
    model = ResNet34(num_classes=args.class_number, T=args.timestep, alpha=args.alpha, imagenet=args.dataset.lower() == 'imagenet')
    functional.set_step_mode(model, step_mode='m')
    functional.set_backend(model, 'cupy', neuron.LIFNode)
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
        logger.info(f'Create model: \n{model}\n')
    ###########################################################################################

    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch]()

    #  Set device
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        logger.warning('using CPU, this will be slow')
    elif args.distributed:
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
                logger.info('Using DistributedDataParallel (DDP)')
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            logger.info('Using DataParallel (DP)')

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

################################ Loss Optimizer Scheduler #################################
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion_train = torch.nn.CrossEntropyLoss().to(device)
    criterion_val = torch.nn.CrossEntropyLoss().to(device)

    optimizer = create_optimizer(args, model)
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
        logger.info(f'Using optimizer: {optimizer}')

    if args.scheduler == 'cosineLR':
        from timm.scheduler.cosine_lr import CosineLRScheduler
        scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, lr_min=args.lr_min,
                                      warmup_t=args.warmup_t, warmup_lr_init=args.warmup_lr_init)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            logger.info('Using CosineLRScheduler')
###########################################################################################

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    ################################# Data loading code #################################
    if args.dummy:
        logger.info("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        train_dataset, val_dataset = get_dataset(args)

    mixup = v2.MixUp(num_classes=args.class_number)
    cutmix = v2.CutMix(num_classes=args.class_number)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    ########################################################################################

    if args.evaluate:
        result = validate(val_loader, model, criterion_val, args)
        print(result)
        return result

    ################################# Training loop #################################
    best_acc1 = 0.
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    for epoch in range(args.start_epoch, args.epochs):

        start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # train for one epoch
        train(train_loader, model, criterion_train, optimizer, epoch, device, args, scaler, cutmix_or_mixup)

        # evaluate on validation set
        acc1, acc5, losses = validate(val_loader, model, criterion_val, args, epoch)
        
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        writer.add_scalar('Train/lr', lr, epoch)
        writer.add_scalar('Val/MaxAcc', best_acc1, epoch)
        # remember best acc@1 and save checkpoint
        
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
            }, is_best, epoch, args, weights_path, acc1, acc5, losses)
        
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            logger.info(f'''epoch={epoch+1}, val_acc1={acc1}, val_acc5={acc5}, total_time={(time.time() - start_time):.4f}, LR={lr:.8f}\n''')
    ################################# Training loop #################################


# In[ ]:


def train(train_loader, model, criterion, optimizer, epoch, device, args, scaler, cutmix_or_mixup):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch+1))

    step_count = len(train_loader) // args.print_freq + 1
    step_temp = 0
    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if (args.t_train>0) and (args.t_train<args.timestep):
            sec_list = np.random.choice(images.shape[1], args.t_train, replace=False)
            sec_list.sort()
            images = images[:, sec_list]

        if (epoch<=args.epochs*2/3) and args.mixup:
            images, target_for_loss = cutmix_or_mixup(images, target)
        else:
            target_for_loss = target
        # compute output
        if args.amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output = model(images)
                loss = criterion(output, target_for_loss)
        else:
            output = model(images)
            loss = criterion(output, target_for_loss)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, args.topk))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(model)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i+1)
            writer.add_scalars('Train/Acc', {'Top1':top1.avg, 'Top5':top5.avg}, step_count*epoch+step_temp)
            writer.add_scalar('Train/Loss', losses.avg, step_count*epoch+step_temp)
            step_temp += 1
    progress.display(i+1)
    writer.add_scalars('Train/Acc', {'Top1':top1.avg, 'Top5':top5.avg}, step_count*epoch+step_temp)
    writer.add_scalar('Train/Loss', losses.avg, step_count*epoch+step_temp)


# In[ ]:


def validate(val_loader, model, criterion, args, epoch=None):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(tqdm(loader)):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, args.topk))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                functional.reset_net(model)
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
            progress.display(i+1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()
    writer.add_scalars('Val/Acc', {'Top1':top1.avg, 'Top5':top5.avg}, epoch)
    writer.add_scalar('Val/Loss', losses.avg, epoch)

    return top1.avg, top5.avg, losses.avg


# In[ ]:


if __name__ == '__main__':
    main()