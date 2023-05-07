import argparse
import os
import sys
import time
from pathlib import Path
from contextlib import suppress
import shutil

import pandas as pd 
import numpy as np
import math

import torch
import torchvision.models as models
from patches import Patches
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision import models as models
from torchvision import transforms
import copy 

torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--best',type=str, default= '/vol/research/prunepath/prune/fold2/best/', help='path to save best models')
    parser.add_argument('--pruned',type=str, default= '/vol/research/prunepath/prune/fold2/pruned/', help='path to save pruned models')
    parser.add_argument('--root_path', type=str, default='/vol/research/SF_datasets/radiology/necrosis/single_20/fold2/', help='Root path containing dataset')
    parser.add_argument('--phase',type=str, default= 'train', help='path to save best models')
    parser.add_argument('--num_classes', type=int, default= 2, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--lr', type=float, default= 0.001 ,help='Learning rate')
    parser.add_argument('--ft_pruned',type=str, default= '/vol/research/prunepath/prune/fold2/ft_pruned/', help='path to save ft pruned models')
    args = parser.parse_args()
    fine_tuning_pruned_model(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_checkpoint(checkpoint, prune_percentage, pruned_dir):
    filename = pruned_dir + '/' + str(prune_percentage) + '.pt'
    torch.save(checkpoint, filename)
        


def fine_tuning_pruned_model(args):
   
    pruning_percentages = [5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 90]
    
    for percentage in pruning_percentages:    # Load the saved model
        model_path = args.pruned + str(percentage) + '/' + str(percentage) + '.pt'
        best = Path(model_path)
        # best.mkdir(exist_ok=True, parents=True)

        model = models.densenet161(pretrained=False)
        # for param in model.parameters():
        #     param.requires_grad = False
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs,2)
        model.to(device)
        
        bestmodel = torch.load(best)
        model.load_state_dict(bestmodel['classifier'])

        # Define the new linear layer
        num_classes = 2  # number of output classes
        new_layer = torch.nn.Linear(in_features=model.classifier.in_features, out_features=num_classes)

        # Replace the old linear layer with the new one
        model.classifier = new_layer
        
        # Define the optimizer
        class_weights = torch.FloatTensor([0.27, 0.73]).to(device)               # change weights according to fold 
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Load the dataset
        datasets_dict = {phase: Patches(root= args.root_path + phase, phase = phase) for phase in ['train', 'val']}
        dataloaders_dict = {phase: torch.utils.data.DataLoader(datasets_dict[phase], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) for phase in ['train', 'val']}

        num_epochs = 30
        model = retrain_model(model, dataloaders_dict, criterion, optimizer, num_epochs)
    
        checkpoint = {     
            'classifier': model.state_dict(),
        }
        save_checkpoint(checkpoint, percentage, args.ft_pruned)



def retrain_model(model, dataloader, criterion, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0

        for i, (inputs, paths10x, labels) in enumerate(dataloader['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            paths = paths10x

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        train_losses = running_loss / len(dataloader['train'])
        training_accuracy = running_corrects/ len(dataloader['train'])

        print("Epoch : {} Train Loss = {:.6f}, Train Accuracy = {:.6f}".format(epoch, train_losses, training_accuracy))
    return model



if __name__ == '__main__':
    main()

