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
from patches_test import Patches
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
    parser.add_argument('--best',type=str, default= '/vol/research/prunepath/prune/fold3.1/best/', help='path to save best models')
    parser.add_argument('--pruned',type=str, default= '/vol/research/prunepath/prune/fold3.1/pruned/', help='path to save pruned models')
    args = parser.parse_args()
    create_pruned_model(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_best_model(path, reverse=False, suffix='.pt'):
    """Load latest checkpoint from target directory. Return None if no checkpoints are found."""
    path, file = Path(path), None
    files = (f for f in sorted(path.iterdir(), reverse=not reverse) if f.suffix == suffix)
    with suppress(StopIteration):
        file = next(f for f in files)
    return file

def save_checkpoint(checkpoint, prune_percentage, pruned_dir):
    filename = pruned_dir + '/' + str(prune_percentage) + '.pt'
    torch.save(checkpoint, filename)

def pruning(model, pruning_percentage):
    model1 = copy.deepcopy(model)
    length = len(list(model1.parameters()))
    for i, param in enumerate(model1.parameters()):
        if len(param.size())!=1 and i<length-2:
            weight = param.detach().cpu().numpy()
            weight[np.abs(weight)<np.percentile(np.abs(weight), pruning_percentage)] = 0
            weight = torch.from_numpy(weight).to(device)
            param.data = weight
    return model1

def create_pruned_model(args):

    best = Path(args.best)
    best.mkdir(exist_ok=True, parents=True)

    model = models.densenet161(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs,2)
    model.to(device)

    bestmodel = get_best_model(best)
    bestmodel = torch.load(bestmodel)
    model.load_state_dict(bestmodel['classifier'])
    
    # for multiple pruning percentages 
    pruning_percentages = [5, 10, 15, 20, 25, 30, 35, 40]


    for percentage in pruning_percentages:
        model = pruning(model, percentage)
        model.eval()
        checkpoint = {     
            'classifier': model.state_dict(),
        }
        save_checkpoint(checkpoint, percentage, args.pruned)


    return model


if __name__ == '__main__':
    main()
    
    