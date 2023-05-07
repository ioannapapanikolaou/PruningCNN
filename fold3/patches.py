import os
import glob
import numpy as np
import pandas as pd 
import random
from PIL import Image
from torch.utils import data
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T



class Patches(data.Dataset):
    def __init__(self, root, phase):
        self.phase = phase

        imgs = []
        if phase == 'train':
            for path, subdirs, files in os.walk(root):
                for fn in files:
                    if fn.endswith(".png") == True:
                        imgs.append(path + "/" + fn)

            self.imgs = imgs


        elif phase == 'val' or 'test':
            for path, subdirs, files in os.walk(root):
                for fn in files:
                    if fn.endswith(".png") == True:
                        imgs.append(path + "/" + fn)

            self.imgs = imgs


        if self.phase == 'train':
            self.transforms = T.Compose([T.Resize(224),T.RandomHorizontalFlip(),
                                    T.RandomVerticalFlip(),
                                    T.ColorJitter(brightness=0.5, hue=0.4),T.ToTensor(), T.Normalize([0.676, 0.472, 0.715], [0.147, 0.158, 0.108])]) # https://discuss.pytorch.org/t/normalization-image-for-using-pretrain-model/15348

        else:
            self.transforms = T.Compose([T.Resize(224),T.ToTensor(), T.Normalize([0.676, 0.472, 0.715], [0.147, 0.158, 0.108])])


    def __getitem__(self, index):
        if self.phase == 'train':
            path = self.imgs[index]
            data = Image.open(path).convert('RGB')
            data = self.transforms(data)
            label = int(path.split('/')[9])


        elif self.phase == 'val' or 'test':
            path = self.imgs[index]
            data = Image.open(path).convert('RGB')
            data = self.transforms(data)

            label = int(path.split('/')[9])


        return data, path, label


    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    root = 'data/'
    train_dataset = Patches(root=root+ "train/", phase= 'train')
    trainloader = data.DataLoader(train_dataset, batch_size=1)
    print(len(trainloader))
    paths = []
    predictions = []
    for i, (data, path, label) in enumerate(trainloader):

        print(path)


        
# https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d



