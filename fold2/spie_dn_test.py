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
import time


torch.backends.cudnn.deterministic = True


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default= 2,
                       help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch Size')
    parser.add_argument('--root_path', type=str, default='/Users/taran/Downloads/dn_data2/',
                        help='Root path containing dataset')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers')
    parser.add_argument('--csv_path', type=str, default='/Users/taran/Desktop/test_soft.csv',
                        help='Number of workers')
    parser.add_argument('--best',type=str, default= '/Users/taran/Desktop/dn10x_best/', help='path to save best models')
    parser.add_argument('--phase',type=str, default= 'val', help='path to save best models')
    args = parser.parse_args()
    twomags_bn(args)
    time_elapsed = time.time() - start_time
    print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





def get_best_model(path, reverse=False, suffix='.pt'):
    """Load latest checkpoint from target directory. Return None if no checkpoints are found."""
    path, file = Path(path), None
    files = (f for f in sorted(path.iterdir(), reverse=not reverse) if f.suffix == suffix)
    with suppress(StopIteration):
        file = next(f for f in files)
    return file



def twomags_bn(args):

    best = Path(args.best)
    best.mkdir(exist_ok=True, parents=True)

    datasets_dict = {phase: Patches(root= args.root_path + phase, phase = phase) for phase in [args.phase]}
    dataloaders_dict = {phase: torch.utils.data.DataLoader(datasets_dict[phase], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) for phase in [args.phase]}


    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html convnet as a feaure extractor

    model = models.densenet161(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs,2)
    model.to(device)


    bestmodel = get_best_model(best)
    bestmodel = torch.load(bestmodel)
    model.load_state_dict(bestmodel['classifier'])
   
    model.eval()

    predictions = []
    imgpaths = []
    probabilities = []
    true_labels = [] 
    topclass = []

    with torch.no_grad():
        for i,(inputs10x, paths10x, labels) in enumerate(dataloaders_dict[args.phase]):

            inputs10x = inputs10x.to(device)
            test_labels = labels.to(device)
            test_paths = paths10x

            # Forward pass to get output/logits
            outputs =  model(inputs10x)
            _, pred = torch.max(outputs, 1) #pred = torch.max(outputs, 1)[1]

            # ASK ABOUT PROBABILTIIES 
            prob = F.softmax(outputs, dim=1)
            
            top_p, top_class = prob.topk(1, dim = 1)

            for x in pred:
                predictions.append(x.cpu().numpy().item())

            for j in top_class:
                topclass.append(j.cpu().numpy().item())

            for q in top_p:
                probabilities.append(q.cpu().numpy().item())

            
            for z in test_paths:
                imgpaths.append(z)
            
            for l in test_labels:
                true_labels.append(l.cpu().numpy().item())


    b = {'paths': imgpaths, 'true_labels': true_labels, 'probabilities': probabilities, 'topclass': topclass, 'predicted_labels': predictions}
    df = pd.DataFrame.from_dict(b)
    df.to_csv(args.csv_path)
    



if __name__ == '__main__':
    main()


