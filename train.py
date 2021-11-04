import time
import os
import os.path as osp
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def train(loader, net, optimizer, criterion, device):
    net.train()
    epoch_loss = 0.
    epoch_corrects = 0.

    for ind, (img_tensor, label, _) in enumerate(loader):
        img_tensor, label = img_tensor.to(device), label.to(device)
        output = net(img_tensor)
        loss = criterion(output, label)

        _, pred = output.max(dim=1)
        epoch_corrects += torch.sum(pred == label.data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()*img_tensor.size(0)

    return epoch_corrects, epoch_loss


def val(loader, net, optimizer, criterion, device):
    with torch.no_grad():
        net.eval()

        epoch_loss = 0.
        epoch_corrects = 0.
    
        for ind, (img_tensor, label, _) in enumerate(loader):
            img_tensor, label = img_tensor.to(device), label.to(device)
            output = net(img_tensor)
            loss = criterion(output, label)
    
            _, pred = output.max(dim=1)
            epoch_corrects += torch.sum(pred == label.data)
    
            epoch_loss += loss.item()*img_tensor.size(0)
    
        return epoch_corrects, epoch_loss


def fit(net_type, loaders:dict, ds_size:dict, log_dir, net, epochs, device='cuda:0'):
    checkpoint_dir = osp.join('checkpoints', net_type)
    # log_dir = osp.join('tb_logs', net_type)

    if osp.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir)
    if osp.isdir(log_dir):
        shutil.rmtree(log_dir)

    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    # lr = 0.01 if net_type == 'pretrained_resnet50' else 0.01
    lr = 0.01
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7)
    # milestones = [1,60]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,60], gamma=0.1, last_epoch=-1)

    train_loader = loaders.get('train')
    val_loader = loaders.get('val')
    train_ds_size = ds_size.get('train')
    val_ds_size = ds_size.get('val')

    # tensorboard
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    # time of we run the script
    TIME_NOW = datetime.now().strftime(DATE_FORMAT)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, TIME_NOW))

    since = time.time()
    print('Start Training...')
    print('=='*40)

    for epoch in range(epochs):
        print(f'Epoch {epoch}, lr: {lr:.4f}')
        # train
        epoch_corrects, epoch_loss = train(train_loader, net, optimizer, criterion, device)
        train_accuracy, train_loss = epoch_corrects / train_ds_size, epoch_loss / train_ds_size
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('accuracy/train', train_accuracy, epoch)
        print(f'train - loss: {train_loss:.4f}, accuracy: {train_accuracy:.4f}')

        # save checkpoint
        checkpoint_filename = '{0:0>3}.pth'.format(epoch)
        checkpoint_path = osp.join(checkpoint_dir, checkpoint_filename)
        torch.save({
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }, checkpoint_path)

        # val
        epoch_corrects, epoch_loss = val(val_loader, net, optimizer, criterion, device)
        val_accuracy, val_loss = epoch_corrects / val_ds_size, epoch_loss / val_ds_size
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('accuracy/val', val_accuracy, epoch)
        print(f'val - loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}')

        # optimizer.step()
        # scheduler.step(val_loss)
        # lr = scheduler.get_lr()
        scheduler.step(val_loss)
        writer.add_scalar('learning rate', lr, epoch)

        # time_elapsed = int(time.time() - since)
        # print('CLASSIFIER TRAINING TIME {}m {}s'.format(time_elapsed//60, time_elapsed%60))
        # print('=='*40)
    # print()

    return train_accuracy, train_loss, val_accuracy, val_loss
