# Helper functions to load data for in-distribution and out-of-distribution datasets
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

from PIL import Image
import os
import os.path
import numpy as np
import scipy.io as sio
import random
import pandas as pd
import medmnist
from torch.utils.data import Dataset


class Modified_medmnist(Dataset):

    def __init__(self,
                 split,
                 transform=None,
                 target_transform=None,
                 data_path='../data/ChestMnist/',
                 chunk= 0,
                 as_rgb=True,
                 root=None,
                image_size = 32): 
        super().__init__()
        npzfile = np.load(data_path+split+'/chunk_'+str(chunk)+'.npz')
        self.data = npzfile['x']
        # print('labels', self.labels.tolist())
        self.targets = npzfile['y']
        print('IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII',self.targets.shape)
        # print('labels', self.targets)

        self.view_classes = [0,1]

        self.image_size = image_size
        self.as_rgb = as_rgb
        self.transform = transform
        self.target_transform = target_transform

        
    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index].astype(int)
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print('here',img.max(),img.min(),target)
        return img, target

    def __len__(self):
        return len(self.data)