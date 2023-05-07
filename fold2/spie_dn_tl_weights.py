import argparse
import os
import sys
import time
from pathlib import Path
from contextlib import suppress
import shutil

import numpy as np
import math

import torch
import torchvision.models as models
from patches import Patches
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision import models as models
from torchvision import transforms
from torch.autograd import Variable

torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default= 30,
                       help='Number of epochs')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--num_classes', type=int, default= 2,
                       help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch Size')
    parser.add_argument('--root_path', type=str, default='/Users/taran/Downloads/dn_data2/',
                        help='Root path containing dataset')
    parser.add_argument('--lr', type=float, default= 0.01 ,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers')
    parser.add_argument('--scheduler_step', type=int, default=10, help='Learning rate scheduler step.')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='Learning rate scheduler gamma.')
    parser.add_argument('--checkpoint',type=str, default= '/Users/taran/Desktop/dnmodels2/', help='path to save models')
    parser.add_argument('--valid_loss',type=str, default= '/Users/taran/Desktop/loss2/', help='path to save models')
    parser.add_argument('--best',type=str, default= '/Users/taran/Desktop/dn10x_best2/', help='path to save best models')

    args = parser.parse_args()
    twomags_bn(args)


def save_checkpoint(checkpoint, is_best, epoch, ckpt_dir, best_dir):
    filename = ckpt_dir + '/' + str(epoch) + '.pt'
    torch.save(checkpoint, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        shutil.copyfile(filename, best_dir + '/'+ 'model_best.pt')


def get_latest_ckpt(path):
    pathing = os.listdir(path)
    if len(pathing)>0:
        filelist = sorted(pathing,key=lambda x: int(os.path.splitext(x)[0]))
        file_name = str(path) + '/' + filelist[-1]
    else:
        file_name = None
    
    return file_name

def store_valid_loss(checkpoint, ckpt_dir):
    filename = ckpt_dir + '/' + 'valid_loss'+ '.pt'
    torch.save(checkpoint, filename)



def get_loss(path):
    checkpoint  = torch.load(path)
    return checkpoint




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def twomags_bn(args):

    start_epoch = args.start_epoch
    start_time = time.time()

    checkpoint = Path(args.checkpoint)
    checkpoint.mkdir(exist_ok=True, parents=True)

    checkpoint_dir = checkpoint
    checkpoint = get_latest_ckpt(checkpoint_dir)

    if not os.path.exists(args.valid_loss):
        os.makedirs(args.valid_loss)


    best = Path(args.best)
    best.mkdir(exist_ok=True, parents=True)


    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html convnet as a feaure extractor

    model = models.densenet161(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier.in_features
    
    model.classifier = torch.nn.Linear(num_ftrs,2)
    model.to(device)
    

    datasets_dict = {phase: Patches(root= args.root_path + phase, phase = phase) for phase in ['train', 'val']}
    dataloaders_dict = {phase: torch.utils.data.DataLoader(datasets_dict[phase], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) for phase in ['train', 'val']}
    
    class_weights = torch.FloatTensor([0.27, 0.73]).to(device) # change weights according to fold 
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    valid_loss_min = 100000
    if checkpoint is not None:
        print("Checkpoint Detected...")
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['classifier'])
        criterion.load_state_dict(checkpoint['criterion'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.start_epoch = checkpoint['epoch']
    
    if os.path.exists(args.valid_loss + '/' + 'valid_loss.pt'):
        loss_checkpoint = get_loss(args.valid_loss + '/' + 'valid_loss.pt')
        valid_loss_min = loss_checkpoint['valid_loss_min']


    for epoch in range(args.start_epoch, args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs))
        print('-' * 60)

        total_loss = 0
        train_accuracy = 0
        model.train()

        for i, (inputs10x, paths10x, labels) in enumerate(dataloaders_dict['train']):

            inputs10x = inputs10x.to(device)
            labels = labels.to(device)
            paths = paths10x

            optimizer.zero_grad()

            outputs = model(inputs10x)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, pred = torch.max(outputs, 1)
            equality_check = (labels.data == pred)
            train_accuracy += equality_check.type(torch.FloatTensor).mean()

            # Getting gradients w.r.t. parameters
            loss.backward()
            optimizer.step()

        scheduler.step()

        train_losses = total_loss / len(dataloaders_dict['train'])
        training_accuracy = train_accuracy/ len(dataloaders_dict['train'])

        print("Epoch : {} Train Loss = {:.6f}, Train Accuracy = {:.6f}".format(epoch, train_losses, training_accuracy))


        model.eval()
        val_loss = 0
        val_accuracy = 0

        with torch.no_grad(): # Tell torch not to calculate gradients
            for i, (inputs10x, paths10x, labels) in enumerate(dataloaders_dict['val']):
                val_inputs10x = inputs10x.to(device)
                val_labels = labels.to(device)
                val_paths = paths10x

                val_outputs = model(val_inputs10x)

                val_loss += criterion(val_outputs, val_labels).item()

                _val, val_pred = torch.max(val_outputs, 1)
                val_equality_check = (val_labels.data == val_pred)
                val_accuracy += val_equality_check.type(torch.FloatTensor).mean()


        val_losses = val_loss / len(dataloaders_dict['val'])
        validation_accuracy = val_accuracy/ len(dataloaders_dict['val'])

     

        print("Epoch : {} Validation Loss = {:.6f}, Validation Accuracy = {:.6f}".format(epoch, val_losses, validation_accuracy))

        checkpoint = {
            'epoch': epoch,
            'classifier': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'criterion': criterion.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        save_checkpoint(checkpoint, False, epoch, args.checkpoint, args.best)

        # save the model if validation loss has decreased
        if val_losses <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, val_losses))
            print('-' * 60)
            # save checkpoint as best model
            save_checkpoint(checkpoint, True, epoch, args.checkpoint, args.best)
        
            valid_loss_min = val_losses
            val_checkpoint = {'valid_loss_min': valid_loss_min}
            store_valid_loss(val_checkpoint, args.valid_loss)


    time_elapsed = time.time() - start_time
    print('-' * 60)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('-' * 60)

if __name__ == '__main__':
    main()



