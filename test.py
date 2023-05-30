# -*- coding: utf-8 -*-
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
import models.moe as moe
import attacks
from utils import get_logger

parser = argparse.ArgumentParser(description='Test PGD attack cifar',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--data_dir', type=str, default='./data', help='dataset dir path')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--arch', type=str, default='Moe1',
                    choices=['Moe1', 'Lorot', 'Nomoe', 'Moe1flip', 'Moe1sc'], help='Choose architecture.')
# PGD
parser.add_argument('--pgd_type', default=1, type=int, choices=[1,2], 
                    help='choose k step for PGD attack (1: 20 step, 2: 100 step)')

# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--load', '-l', type=str, default=None,
                    help='Checkpoint path to resume / test.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu', type=int, default=None, help='use id gpu')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()

if not os.path.exists(args.load):
    raise Exception('load path is not found')
args.save = os.path.dirname(args.load)

logger = get_logger('PGD-Test', args.save)

if args.pgd_type == 1:
    args.num_steps = 20
    args.step_size = 2./255
elif args.pgd_type == 2:
    args.num_steps = 100
    args.step_size = 0.3/255
else:
    raise Exception('Please choose PGD type 1 or 2, see help')

logger.info(f'Argument: {args}')

torch.manual_seed(1)
np.random.seed(1)

# # mean and standard deviation of channels of CIFAR-10 images
# mean = [x / 255 for x in [125.3, 123.0, 113.9]]
# std = [x / 255 for x in [63.0, 62.1, 66.7]]

# train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
#                                trn.ToTensor()])
test_transform = trn.Compose([trn.ToTensor()])

if args.dataset == 'cifar10':
    # train_data = dset.CIFAR10('/share/data/vision-greg/cifarpy', train=True, transform=train_transform)
    test_data = dset.CIFAR10(args.data_dir, train=False, transform=test_transform)
    args.num_classes = 10
else:
    # train_data = dset.CIFAR100('/share/data/vision-greg/cifarpy', train=True, transform=train_transform)
    test_data = dset.CIFAR100(args.data_dir, train=False, transform=test_transform)
    args.num_classes = 100


# train_loader = torch.utils.data.DataLoader(
#     train_data, batch_size=args.batch_size, shuffle=True,
#     num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)
# Create model
# if 'allconv' in args.model:
#     net = AllConvNet(num_classes)
# else:
#     net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
net = moe.__dict__[args.arch](depth=args.layers, 
                              widen_factor=args.widen_factor,
                              drop_rate=args.droprate,
                              num_classes=args.num_classes)
logger.info(f'model_arch: {args.arch}')

# if args.ngpu > 0:
#     net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

# Restore model if desired
if args.load is not None:
    checkpoint = torch.load(args.load)
    net.load_state_dict(checkpoint['model'])
    logger.info(f"Check Point Loading: model is LOADED")
    logger.info(f'Epoch: {checkpoint["epoch"]}')

#if args.ngpu > 1:
#    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    if args.gpu is not None:
        logger.warning(f"USE GPU: {args.gpu} for testing")
    net.cuda(args.gpu)
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders


adversary = attacks.PGD(epsilon=8./255, num_steps=args.num_steps, step_size=args.step_size, attack_rotations=False).cuda(args.gpu)


def evaluate(adv=True):
    net.eval()
    if adv is False:
        torch.set_grad_enabled(False)
    running_loss = 0
    running_acc = 0
    count = 0
    for i, batch in enumerate(test_loader):
        bx = batch[0].cuda(args.gpu)
        by = batch[1].cuda(args.gpu)

        count += by.size(0)

        adv_bx = adversary(net, bx, by, None, None, None) if adv else bx
        with torch.no_grad():
            logits = net(adv_bx * 2 - 1)

        loss = F.cross_entropy(logits.data, by, reduction='sum')
        running_loss += loss.cpu().data.numpy()
        running_acc += (torch.max(logits, dim=1)[1] == by).float().sum(0).cpu().data.numpy()
    running_loss /= count
    running_acc /= count

    loss = running_loss
    acc = running_acc

    if adv is False:
        torch.set_grad_enabled(True)
    return loss, acc


loss, acc = evaluate(adv=False)
logger.info('Normal Test Loss: {:.4f} | Normal Test Acc: {:.4f}'.format(loss, acc))
loss, acc = evaluate(adv=True)
logger.info('Adv Test Loss: {:.4f} | Adv Test Acc: {:.4f}'.format(loss, acc))