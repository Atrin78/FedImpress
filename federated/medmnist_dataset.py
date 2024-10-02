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
from numpy import inf
from medmnist import BloodMNIST



class Modified_medmnist(Dataset):

    def __init__(self,
                 split,
                 transform=None,
                 target_transform=None,
                 data_path='../data/Bloodmnist/',
                 chunk= 0,
                 as_rgb=True,
                 root=None,
                image_size = 32,
                mean = None,
                std = None): 
        super().__init__()
        npzfile = np.load(data_path+split+'/chunk_'+str(chunk)+'.npz')
        self.images = npzfile['x']
        # print('labels', self.labels.tolist())
        if split == 'cifar10' or split == 'VHL':
            self.labels = npzfile['y']
        else:
            self.labels = npzfile['y'].squeeze()
        self.synthesized = False
        if split == 'VHL':
            self.synthesized = True
        self.image_size = image_size
        if split == 'test' or (mean != None and std != None):
            self.mean, self.std = mean, std
        elif split == 'VHL':
            pass
        else:
            self.mean, self.std = self.compute_mean_std()
    
        # train_dataset = BloodMNIST(split=split , transform=None, download=False,as_rgb= True)
        # self.images = train_dataset.imgs
        # self.labels = train_dataset.labels.squeeze()
       
        # print(self.labels,self.labels.shape)
        print('IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII',self.print_number_of_samples())
        # print('labels', self.targets)

        self.view_classes = [0,1,2,3,4,5,6,7]

        self.as_rgb = as_rgb
        if split != 'VHL':
            self.transform = self.get_transform(True)
            self.transform_synthesized = self.get_synthesized_transform(True)
        self.transform_synthesized = None
        self.target_transform = target_transform

        
    def __getitem__(self, index):

        img, target = self.images[index], self.labels[index]

        if not self.synthesized:
            img = Image.fromarray(img)
            if self.as_rgb:
                img = img.convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            if self.transform_synthesized is not None:
                img = torch.tensor(img)
                # img =   self.transform_synthesized(img)
            target = torch.tensor(target)
        # print('here',img.max(),img.min(),target)
        return img, target

    def __len__(self):
        return len(self.labels)
    
    def compute_class_weights(self):
        
        
        freq = np.bincount((self.labels).astype(int),minlength = 8)
        # freq= np.concatenate((freq_proto,freq))
        
        freq = freq/freq.max()
        # print('freeeq',freq)
        total_samples = len(self.labels)
        inv_freq = 1. / freq
        inv_freq[inv_freq == inf] = 0
        # print('freeeq',inv_freq)
        #print(total_samples * inv_freq / len(np.unique(arr)) )
        return  torch.tensor(inv_freq).type(torch.cuda.FloatTensor)
    
    def print_number_of_samples(self):
        
        
        freq = np.bincount((self.labels).astype(int),minlength = 8)
        
        
        print('number_of_samples:',freq)
        #print(total_samples * inv_freq / len(np.unique(arr)) )
        return  
    
    def compute_mean_std(self):
        mean = [0,0,0]
        std = [1,1,1]
        sum_m = 0
        images = torch.zeros((self.images.shape[0],3,self.image_size,self.image_size))
        print(self.images.shape)
        if not self.synthesized:
            transform = self.get_transform(False)
            
        else:
            transform = self.get_synthesized_transform(False)
        for i in range(len(self.images)):
            if not self.synthesized:
                images[i] = transform(Image.fromarray(self.images[i]))
            else:
                images[i] = torch.tensor(self.images[i])
                # images[i] = transform(self.images[i])

        mean = torch.mean(images,dim=(0,2,3))
        std = torch.std(images,dim=(0,2,3))
        # print(len(mean), len(std))
        return mean, std
        
        
    def get_transform(self,mean):
        if mean:
            return transforms.Compose(
                [
                 transforms.CenterCrop(256),
                 transforms.ToTensor(),
                 transforms.Normalize(self.mean, self.std)
                ])
        else:
            return transforms.Compose(
                [
                 transforms.CenterCrop(256),
                 transforms.ToTensor()
                ])
    def get_synthesized_transform(self,mean):
        if mean:
            return transforms.Compose(
                [transforms.Normalize(self.mean, self.std)
                ])
        else:
            return transforms.Compose(
                [transforms.ToTensor()
                ])