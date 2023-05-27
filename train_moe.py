import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.allconv import AllConvNet
from models.wrn import WideResNet
import models.moe as moe
import attacks

from utils import SslTransform, ConfigObj, get_logger, TBLog

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

def save_checkpoint(model, epoch, optimizer, scheduler, name, path, logger):
    path = os.path.join(path, name)
    model_save = model.module if hasattr(model, 'module') else model
    torch.save({'model': model_save.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},path)
    logger.info(f"model saved: {path}")

def train(net, train_loader, adversary, scheduler, optimizer, args):
    net.train()  # enter train mode
    loss_avg = 0.0
    for (bx, ssl_label), by in train_loader:
        curr_batch_size = bx.size(0)

        # by_prime = torch.cat((torch.zeros(bx.size(0)), torch.ones(bx.size(0)),
        #                       2*torch.ones(bx.size(0)), 3*torch.ones(bx.size(0))), 0).long()
        # bx = bx.numpy()
        # use torch.rot90 in later versions of pytorch
        # bx = np.concatenate((bx, bx, np.rot90(bx, 1, axes=(2, 3)),
                            #  np.rot90(bx, 2, axes=(2, 3)), np.rot90(bx, 3, axes=(2, 3))), 0)
        # bx = torch.FloatTensor(bx)
        bx, by = bx.cuda(args.gpu), by.cuda(args.gpu)
        ssl_label = torch.stack(ssl_label).cuda(args.gpu) if isinstance(ssl_label, (tuple, list)) else ssl_label.cuda(args.gpu)

        adv_bx = adversary(net, bx, by, ssl_label, None, None)

        # forward
        # logits, = net(adv_bx * 2 - 1)

        if args.arch == "Moe1": # use rot flip sc
            sup_output, rot_output, flip_output, sc_output, gate_output = net(adv_bx * 2 - 1)
            gate = F.softmax(gate_output, dim=1).mean(dim=0)
            ssl_loss = (F.cross_entropy(rot_output, ssl_label[0], reduction='mean') * gate[0].item()+
                        F.cross_entropy(flip_output, ssl_label[1], reduction='mean') * gate[1].item()+
                        F.cross_entropy(sc_output, ssl_label[2], reduction='mean') * gate[2].item())
            
        if args.arch == "Moe1flip": # use rot flip sc
            sup_output, rot_output, flip_output, gate_output = net(adv_bx * 2 - 1)
            gate = F.softmax(gate_output, dim=1).mean(dim=0)
            ssl_loss = (F.cross_entropy(rot_output, ssl_label[0], reduction='mean') * gate[0].item()+
                        F.cross_entropy(flip_output, ssl_label[1], reduction='mean') * gate[1].item())
            
        if args.arch == "Moe1sc": # use rot flip sc
            sup_output, rot_output, sc_output, gate_output = net(adv_bx * 2 - 1)
            gate = F.softmax(gate_output, dim=1).mean(dim=0)
            ssl_loss = (F.cross_entropy(rot_output, ssl_label[0], reduction='mean') * gate[0].item()+
                        F.cross_entropy(sc_output, ssl_label[1], reduction='mean') * gate[1].item())

        elif args.arch == "Nomoe": # use 2 task
            sup_output, rot_output, flip_output, sc_output = net(adv_bx * 2 - 1)
            ssl_loss = (F.cross_entropy(rot_output, ssl_label[0], reduction='mean') +
                        F.cross_entropy(flip_output, ssl_label[1], reduction='mean') +
                        F.cross_entropy(sc_output, ssl_label[2], reduction='mean') )
            
        elif args.arch == "Lorot":
            sup_output, rot_output = net(adv_bx * 2 - 1)
            ssl_loss = F.cross_entropy(rot_output, ssl_label, reduction='mean')

        # backward
        scheduler.step()
        optimizer.zero_grad()
        loss = F.cross_entropy(sup_output, by) + args.ssl_ratio * ssl_loss
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.9 + float(loss) * 0.1

    return loss_avg

def test(net, test_loader, args):
    net.eval()
    loss_avg = 0.0
    correct = 0
    state = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(args.gpu), target.cuda(args.gpu)

            # forward
            output = net(data * 2 - 1)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['val_loss'] = loss_avg / len(test_loader)
    state['val_accuracy'] = correct / len(test_loader.dataset)
    return state


def main(args):
    if os.path.exists(args.save) and not args.overwrite:
        raise Exception('already existing directory: {}'.format(args.save))
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)
    logger = get_logger('PGD-GSSL', args.save)

    train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(),
                               SslTransform(args.arch)])
    test_transform = trn.Compose([trn.ToTensor()])

    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_dir, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_dir, train=False, transform=test_transform, download=True)
        args.num_classes = 10
    else:
        train_data = dset.CIFAR100(args.data_dir, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_dir, train=False, transform=test_transform, download=True)
        args.num_classes = 100
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_bs, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    torch.cuda.set_device(args.gpu)
    # create model
    net = moe.__dict__[args.arch](depth=args.layers, 
                                    widen_factor=args.widen_factor,
                                    drop_rate=args.droprate,
                                    num_classes=args.num_classes)
    start_epoch = 0
    logger.info(f'model_arch: {args.arch}')

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    if args.ngpu > 0:
        if args.gpu is not None:
            logger.warning('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
            logger.warning(f"USE GPU: {args.gpu} for training")
        net.cuda(args.gpu)
        torch.cuda.manual_seed(1)

    optimizer = torch.optim.SGD(
        net.parameters(), args.learning_rate, momentum=args.momentum,
        weight_decay=args.decay, nesterov=True)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))
    
    # resume checkpoint
    if args.load is not None:
        checkpoint = torch.load(args.load, map_location=torch.device('cuda', args.gpu))
        for key in checkpoint.keys():
            if key == 'model':
                net.load_state_dict(checkpoint[key])
            elif key == 'epoch':
                start_epoch = checkpoint[key] + 1
            elif key == 'optimizer':
                optimizer.load_state_dict(checkpoint[key])
            elif key == 'scheduler':
                scheduler.load_state_dict(checkpoint[key])
            logger.info(f"Check Point Loading: {key} is LOADED")

    
    
    cudnn.benchmark = True  # fire on all cylinders
    adversary = attacks.PGD(epsilon=args.epsilon, num_steps=args.num_steps, step_size=args.step_size, attack_rotations=False).cuda(args.gpu)
    
    logger.info(f'Arguments: {args}')

    if args.test:
        test_state = test()
        print(test_state)
        return
    
    tb_log = TBLog(args.save, 'tblog')

    best_val_acc = 0
    for epoch in range(start_epoch, args.epochs):
        begin_epoch = time.time()

        train_loss = train(net, train_loader, adversary, scheduler, optimizer, args)
        val_state = test(net, test_loader, args)

        tb_dict={}
        tb_dict['train/loss'] = train_loss
        tb_dict['eval/loss'] = val_state['val_loss']
        tb_dict['eval/acc'] = val_state['val_accuracy']
        tb_log.update(tb_dict, epoch)

        print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Val Loss {3:.3f} | Val Acurracy {4:.2f}%'.format(
            (epoch),
            int(time.time() - begin_epoch),
            train_loss,
            val_state['val_loss'],
            100. * val_state['val_accuracy'])
        )

        if val_state['val_accuracy'] > best_val_acc:
            best_val_acc = val_state['val_accuracy']
            logger.info(f"Epoch: {epoch:3d} Best Val Acc: {100*best_val_acc:.2f}%")
            save_checkpoint(net, epoch, optimizer, scheduler, 'best_model.pt', args.save, logger)
        elif (epoch+1) % args.save_freq == 0 or (epoch+1) == args.epochs:
            logger.info(f'Save Checkpoint at epoch {epoch:3d}')
            save_checkpoint(net, epoch, optimizer, scheduler, 'lastest_model.pt', args.save, logger)
            

def path_correction(path):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)),path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Adversarial Pertubation Training for Gated SSL')
    parser.add_argument('--path', '-p', type=str, help='config path')
    cli_parser = parser.parse_args()

    config = ConfigObj(path_correction('config/gssl_cifar10_default.yaml'), cli_parser.path)
    args = config.get()

    assert args.arch in ['Moe1', 'Lorot', 'Nomoe', 'Moe1flip', 'Moe1sc']
    # set save_name
    args.save_name += f"_{args.arch}_{args.dataset}"
    args.save = os.path.join(args.save,args.save_name)

    main(args)