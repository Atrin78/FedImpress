"""
FedDC v1: global_client distilled data training using GM + global distilled data using GM
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
from torch import nn, optim
import time
import copy
# from fedbn_nets.models import DigitModel
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from utils.data_utils import CustomDataset

# import fedbn_data_utils as data_utils
# from utils import get_network, get_time, TensorDataset, epoch
# from pretraineddataset import PretrainedDataset, GetPretrained
# from condensation import distribution_matching, distribution_matching_bn, gradient_matching, get_initial_normal
from torchvision.utils import save_image
import random
# from loss_fn import Distance_loss

from PIL import Image
import csv
from medmnist_dataset import Modified_medmnist
import math
# import pandas as pd

import wandb

MEANS = [[0.5594, 0.2722, 0.0819], [0.7238, 0.3767, 0.1002], [0.5886, 0.2652, 0.1481], [0.7085, 0.4822, 0.3445]]
STDS = [[0.1378, 0.0958, 0.0343], [0.1001, 0.1057, 0.0503], [0.1147, 0.0937, 0.0461], [0.1663, 0.1541, 0.1066]]

# MEANS = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
# STDS = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]

def get_label(file):
    retVal = {}
    lines = list(csv.reader(open(file)))
    for line in lines[1:]:
        retVal[line[0][:-1]] = line[-2]
    return retVal

def get_label2(file):
    retVal = {}
    lines = list(csv.reader(open(file)))
    for line in lines[1:]:
        retVal[line[1]] = line[2]
    return retVal

class DrishtiDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transform, train=True):
        self.CLASSES = 2
        self.class_dict = {'Normal': 1, 'Glaucomatous': 0}
        self.transform = transform
        if train:
            self.file = os.path.join(dataset_path, 'Drishti', 'Training')
        else:
            self.file = os.path.join(dataset_path, 'Drishti', 'Testing')
        self.labels = get_label(os.path.join(dataset_path, 'Drishti', 'Drishti-GS1_diagnosis.csv'))

    def __len__(self):
        return len(os.listdir(self.file))

    def __getitem__(self, index):
        img_name = os.listdir(self.file)[index]
        image_tensor = self.load_image(os.path.join(self.file,img_name))
        label_tensor = torch.tensor(self.class_dict[self.labels[img_name[:-4]]], dtype=torch.long)

        return image_tensor, label_tensor

    def load_image(self, img_path):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        image = Image.open(img_path).convert('RGB')
        # image = image.resize(dim)

        image_tensor = self.transform(image)

        return image_tensor


class RefugeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transform, test=True):
        self.CLASSES = 2
        self.class_dict = {'Normal': 1, 'Glaucomatous': 0}
        self.transform = transform
        if test:
            # Test Dataset
            self.file = os.path.join(dataset_path, 'REFUGE_segmented', 'Test')
            self.labels = get_label2(os.path.join(dataset_path, 'REFUGE_segmented', 'Glaucoma_label_and_Fovea_location.csv'))
        else:
            # Validation Dataset
            self.file = os.path.join(dataset_path, 'REFUGE_segmented', 'Validation')
            self.labels = get_label2(os.path.join(dataset_path, 'REFUGE_segmented', 'Fovea_locations.csv'))

    def __len__(self):
        return len(os.listdir(self.file))

    def __getitem__(self, index):
        img_name = os.listdir(self.file)[index]
        image_tensor = self.load_image(os.path.join(self.file,img_name))
        label_tensor = torch.tensor(int(self.labels[img_name]), dtype=torch.long)

        return image_tensor, label_tensor

    def load_image(self, img_path):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        image = Image.open(img_path).convert('RGB')

        image_tensor = self.transform(image)

        return image_tensor

def prepare_data(args, datasets, public_dataset, im_size):
    # data_base_path = './data/segmented_retina'
    data_base_path = '../data/retina_balanced'
    # transform_train = transforms.Compose([
    #         transforms.Resize([96, 96]),            
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomRotation((-30,30)),
    #         transforms.ToTensor(),
    # ])

    transform_tensor = transforms.Compose([
            transforms.Resize(im_size),
            transforms.ToTensor()
    ])

    transform_unnormalized = transforms.Compose([
            transforms.Resize(im_size),            
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation((-30,30)),
            transforms.ToTensor()
    ])
    
    # Drishti
    transform_drishti = transforms.Compose([
            transforms.Resize(im_size),            
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
            transforms.Normalize(MEANS[0], STDS[0])
    ])

    transform_drishti2 = transforms.Compose([
        # transforms.Resize(im_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation((-30,30)),
        # transforms.ToTensor(),
        transforms.Normalize(MEANS[0], STDS[0])
    ])

    transform_cifar = transforms.Compose(
        [
            # transforms.Resize(im_size),
         # transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # drishti_trainset = DrishtiDataset(dataset_path=data_base_path,transform=transform_drishti)
    # drishti_testset = DrishtiDataset(dataset_path=data_base_path,transform=transform_drishti, train=False)
    drishti_train_path = os.path.join(data_base_path, 'Drishti', 'Training')
    drishti_test_path = os.path.join(data_base_path, 'Drishti', 'Testing')
    # unnormalized_drishti_trainset = ImageFolder(drishti_train_path, transform=transform_unnormalized)
    drishti_trainset = ImageFolder(drishti_train_path, transform=transform_drishti)
    # drishti_virtualset = ImageFolder(drishti_train_path, transform=transform_drishti)
    drishti_testset = ImageFolder(drishti_test_path, transform=transform_drishti)
    # drishti_concated = torch.utils.data.ConcatDataset([drishti_trainset, drishti_testset])
    
    # kaggle
    transform_kaggle = transforms.Compose([
            transforms.Resize(im_size),            
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
            transforms.Normalize(MEANS[1], STDS[1])
    ])
    transform_kaggle2 = transforms.Compose([
        # transforms.Resize(im_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation((-30,30)),
        # transforms.ToTensor(),
        transforms.Normalize(MEANS[1], STDS[1])
    ])
    kaggle_train_path = os.path.join(data_base_path, 'kaggle_arima', 'Training')
    kaggle_test_path = os.path.join(data_base_path, 'kaggle_arima', 'Testing')
    # unnormalized_kaggle_trainset = ImageFolder(kaggle_train_path, transform=transform_unnormalized)
    kaggle_trainset = ImageFolder(kaggle_train_path, transform=transform_kaggle)
    # kaggle_virtualset = ImageFolder(kaggle_train_path, transform=transform_kaggle)
    kaggle_testset = ImageFolder(kaggle_test_path, transform=transform_kaggle)
    # kaggle_concated = ImageFolder(kaggle_train_path, transform=transform_kaggle, target_transform=torch.tensor)
    # print(kaggle_concated.class_to_idx)

    # RIM
    transform_rim = transforms.Compose([
            transforms.Resize(im_size),            
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
            transforms.Normalize(MEANS[2], STDS[2])
    ])
    transform_rim2 = transforms.Compose([
        # transforms.Resize(im_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation((-30,30)),
        # transforms.ToTensor(),
        transforms.Normalize(MEANS[2], STDS[2])
    ])
    # rim_train_path = os.path.join(data_base_path, 'RIM-ONE_DL_images', 'partitioned_by_hospital', 'training_set')
    # rim_test_path = os.path.join(data_base_path, 'RIM-ONE_DL_images', 'partitioned_by_hospital', 'test_set')
    rim_train_path = os.path.join(data_base_path, 'RIM', 'Training')
    rim_test_path = os.path.join(data_base_path, 'RIM', 'Testing')
    # unnormalized_rim_trainset = ImageFolder(rim_train_path, transform=transform_unnormalized)
    rim_trainset = ImageFolder(rim_train_path, transform=transform_rim)
    # rim_virtualset = ImageFolder(rim_train_path, transform=transform_rim)
    rim_testset = ImageFolder(rim_test_path, transform=transform_rim)
    # rim_concated = torch.utils.data.ConcatDataset([rim_trainset,rim_testset])
    # print(rim_trainset.class_to_idx)
    # print(rim_testset.class_to_idx)
    
    # refuge
    transform_refuge = transforms.Compose([
            transforms.Resize(im_size),            
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
            transforms.Normalize(MEANS[3], STDS[3])
    ])
    transform_refuge2 = transforms.Compose([
        # transforms.Resize(im_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation((-30,30)),
        # transforms.ToTensor(),
        transforms.Normalize(MEANS[3], STDS[3])
    ])
    refuge_train_path = os.path.join(data_base_path, 'REFUGE', 'Training')
    refuge_test_path = os.path.join(data_base_path, 'REFUGE', 'Testing')
    # unnormalized_refuge_trainset = ImageFolder(refuge_train_path, transform=transform_unnormalized)
    refuge_trainset = ImageFolder(refuge_train_path, transform=transform_refuge)
    # refuge_virtualset = ImageFolder(refuge_train_path, transform=transform_refuge)
    refuge_testset = ImageFolder(refuge_test_path, transform=transform_refuge)
    # print(refuge_trainset.class_to_idx)
    # refuge_valset = RefugeDataset(data_base_path, transform=transform_refuge, test=False)
    # refuge_testset = RefugeDataset(data_base_path, transform=transform_refuge)
    # refuge_concated = torch.utils.data.ConcatDataset([refuge_trainset,refuge_valset,refuge_testset])
    # print(torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                               download=True, transform=transform_cifar)[0][0].shape)
    cifar_trainset = CustomDataset(torchvision.datasets.CIFAR10(root='./data', train=True,
                                                  download=True, transform=transform_tensor), args.data_size, transform=transform_cifar)
    cifar_virtualset = CustomDataset(torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transform_tensor), args.data_size, transform=transform_cifar)
    cifar_testset = CustomDataset(torchvision.datasets.CIFAR10(root='./data', train=False,
                                                 download=True, transform=transform_tensor), args.data_size, transform=transform_cifar)

    cifar_generateset = Modified_medmnist(data_path="/home/atrinarya/fed/Federated-Learning/data/VHL/", split='cifar10', chunk='1', mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))
    cifar_virtualset = Modified_medmnist(data_path="/home/atrinarya/fed/Federated-Learning/data/VHL/", split='cifar10', chunk='1', mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))

    min_data_len = min(len(drishti_trainset), len(kaggle_trainset), len(rim_trainset), len(refuge_trainset))

    len_datasets = [len(drishti_trainset), len(kaggle_trainset), len(rim_trainset), len(refuge_trainset), len(cifar_trainset)]

    # amazon_valset = torch.utils.data.Subset(amazon_trainset, list(range(len(amazon_trainset)))[-val_len:])
    shuffled_idxes = [list(range(0, len_datasets[idx])) for idx in range(5)]
    for idx in range(len(shuffled_idxes)):
        random.shuffle(shuffled_idxes[idx])
    drishti_trainset = torch.utils.data.Subset(drishti_trainset, shuffled_idxes[0][:min_data_len])

    # caltech_valset = torch.utils.data.Subset(caltech_trainset, list(range(len(caltech_trainset)))[-val_len:])
    kaggle_trainset = torch.utils.data.Subset(kaggle_trainset, shuffled_idxes[1][:min_data_len])

    # dslr_valset = torch.utils.data.Subset(dslr_trainset, list(range(len(dslr_trainset)))[-val_len:])
    rim_trainset = torch.utils.data.Subset(rim_trainset, shuffled_idxes[2][:min_data_len])

    # webcam_valset = torch.utils.data.Subset(webcam_trainset, list(range(len(webcam_trainset)))[-val_len:])
    refuge_trainset = torch.utils.data.Subset(refuge_trainset, shuffled_idxes[3][:min_data_len])

    self_trainset = torch.utils.data.ConcatDataset([torch.utils.data.Subset(drishti_trainset, shuffled_idxes[0][min_data_len:min_data_len+args.data_size//4]), torch.utils.data.Subset(kaggle_trainset, shuffled_idxes[1][min_data_len:min_data_len+args.data_size//4]), torch.utils.data.Subset(rim_trainset, shuffled_idxes[2][min_data_len:min_data_len+args.data_size//4]), torch.utils.data.Subset(refuge_trainset, shuffled_idxes[3][min_data_len:min_data_len+args.data_size//4])])
    self_virtualset = torch.utils.data.ConcatDataset([torch.utils.data.Subset(drishti_trainset, shuffled_idxes[0][min_data_len:min_data_len+args.data_size//4]), torch.utils.data.Subset(kaggle_trainset, shuffled_idxes[1][min_data_len:min_data_len+args.data_size//4]), torch.utils.data.Subset(rim_trainset, shuffled_idxes[2][min_data_len:min_data_len+args.data_size//4]), torch.utils.data.Subset(refuge_trainset, shuffled_idxes[3][min_data_len:min_data_len+args.data_size//4])])

    VHL_virtualset = Modified_medmnist(data_path="/home/atrinarya/fed/Federated-Learning/data/VHL/", split='VHL', chunk='0')
    VHL_generateset = Modified_medmnist(data_path="/home/atrinarya/fed/Federated-Learning/data/VHL/", split='VHL', chunk='0')
    # cifar_trainset = torch.utils.data.Subset(cifar_trainset, shuffled_idxes[4][:min_data_len])


    # dataset_length = len(drishti_concated)
    # dataset_length_client = int(dataset_length * 0.8)
    # split_length = [dataset_length_client, dataset_length-dataset_length_client]
    # drishti_trainset, drishti_testset = torch.utils.data.random_split(
    # dataset=drishti_concated, 
    # lengths=split_length,
    # generator=torch.manual_seed(0)
    # )

    # dataset_length = len(kaggle_concated)
    # dataset_length_client = int(dataset_length * 0.8)
    # split_length = [dataset_length_client, dataset_length-dataset_length_client]
    # kaggle_trainset, kaggle_testset = torch.utils.data.random_split(
    # dataset=kaggle_concated, 
    # lengths=split_length,
    # generator=torch.manual_seed(0)
    # )

    # dataset_length = len(rim_concated)
    # dataset_length_client = int(dataset_length * 0.8)
    # split_length = [dataset_length_client, dataset_length-dataset_length_client]
    # rim_trainset, rim_testset = torch.utils.data.random_split(
    # dataset=rim_concated, 
    # lengths=split_length,
    # generator=torch.manual_seed(0)
    # )

    # dataset_length = len(refuge_concated)
    # dataset_length_client = int(dataset_length * 0.8)
    # split_length = [dataset_length_client, dataset_length-dataset_length_client]
    # refuge_trainset, refuge_testset = torch.utils.data.random_split(
    # dataset=refuge_concated, 
    # lengths=split_length,
    # generator=torch.manual_seed(0)
    # )

    #####################################
    # Drishti_train_loader = torch.utils.data.DataLoader(drishti_trainset, batch_size=args.batch, shuffle=True)
    # Drishti_test_loader = torch.utils.data.DataLoader(drishti_testset, batch_size=args.batch, shuffle=False)
    # 
    # kaggle_train_loader = torch.utils.data.DataLoader(kaggle_trainset, batch_size=args.batch, shuffle=True)
    # kaggle_test_loader = torch.utils.data.DataLoader(kaggle_testset, batch_size=args.batch, shuffle=False)
    # 
    # rim_train_loader = torch.utils.data.DataLoader(rim_trainset, batch_size=args.batch, shuffle=True)
    # rim_test_loader = torch.utils.data.DataLoader(rim_testset, batch_size=args.batch, shuffle=False)
    # 
    # refuge_train_loader = torch.utils.data.DataLoader(refuge_trainset, batch_size=args.batch, shuffle=True)
    # refuge_test_loader = torch.utils.data.DataLoader(refuge_testset, batch_size=args.batch, shuffle=False)
    # 
    # train_loaders = [Drishti_train_loader, kaggle_train_loader, rim_train_loader, refuge_train_loader]
    # test_loaders = [Drishti_test_loader, kaggle_test_loader, rim_test_loader, refuge_test_loader]

    #print('d')
    #print(len(drishti_trainset))

    #drishti_trainset2 = torch.utils.data.Subset(drishti_trainset, range(len(refuge_trainset)))
    #drishti_trainset2.dataset.transform = transforms.Compose([
    #        transforms.Resize(im_size),            
    #        transforms.RandomHorizontalFlip(),
    #        transforms.RandomRotation((-30,30)),
    #        transforms.ToTensor(),
    #        transforms.Normalize(MEANS[0], STDS[0])
    #])
    #drishti_trainset = torch.utils.data.ConcatDataset([drishti_trainset, drishti_trainset2])

    #print(len(kaggle_trainset))

    #kaggle_trainset2 = torch.utils.data.Subset(kaggle_trainset, range(len(refuge_trainset)))
    #kaggle_trainset2.dataset.transform = transforms.Compose([
    #        transforms.Resize(im_size),            
    #        transforms.RandomHorizontalFlip(),
    #        transforms.RandomRotation((-30,30)),
    #        transforms.ToTensor(),
    #        transforms.Normalize(MEANS[1], STDS[1])
    #])
    #kaggle_trainset = torch.utils.data.ConcatDataset([kaggle_trainset, kaggle_trainset2])

    #print(len(rim_trainset))

    #rim_trainset2 = torch.utils.data.Subset(rim_trainset, range(len(refuge_trainset)))
    #rim_trainset2.dataset.transform = transforms.Compose([
    #        transforms.Resize(im_size),            
    #        transforms.RandomHorizontalFlip(),
    #        transforms.RandomRotation((-30,30)),
    #        transforms.ToTensor(),
    #        transforms.Normalize(MEANS[2], STDS[2])
    #])
    #rim_trainset = torch.utils.data.ConcatDataset([rim_trainset, rim_trainset2])
    


    trainsets = []
    virtualsets = []
    testsets = []
    generatsets = []
    for dataset in datasets:
        if dataset == 'drishti':
            trainsets.append(drishti_trainset)
            testsets.append(drishti_testset)
            # if public_dataset == None:
            #     generatsets.append(drishti_trainset)
            #     virtualsets.append(drishti_virtualset)

        elif dataset == 'kaggle':
            trainsets.append(kaggle_trainset)
            testsets.append(kaggle_testset)
            # if public_dataset == None:
            #     generatsets.append(kaggle_trainset)
            #     virtualsets.append(kaggle_virtualset)

        elif dataset == 'rim':
            trainsets.append(rim_trainset)
            testsets.append(rim_testset)
            # if public_dataset == None:
            #     generatsets.append(rim_trainset)
            #     virtualsets.append(rim_virtualset)

        elif dataset == 'refuge':
            trainsets.append(refuge_trainset)
            testsets.append(refuge_testset)
            # if public_dataset == None:
            #     generatsets.append(refuge_trainset)
            #     virtualsets.append(refuge_virtualset)

        elif dataset == 'cifar':
            trainsets.append(cifar_trainset)
            testsets.append(cifar_testset)
            # if public_dataset == None:
            #     generatsets.append(cifar_trainset)
            #     virtualsets.append(cifar_virtualset)

        elif dataset == 'self':
            trainsets.append(self_trainset)
            # if public_dataset == None:
            #     generatsets.append(self_trainset)
            #     virtualsets.append(self_virtualset)

    if public_dataset != None:
        if public_dataset == 'drishti':
            generatsets.append(CustomDataset(ImageFolder(drishti_train_path, transform=transform_tensor), args.data_size, transform=transform_drishti2))
            virtualsets.append(CustomDataset(ImageFolder(drishti_train_path, transform=transform_tensor), args.data_size, transform=transform_drishti2))

        elif public_dataset == 'kaggle':
            generatsets.append(CustomDataset(ImageFolder(kaggle_train_path, transform=transform_tensor), args.data_size, transform=transform_kaggle2))
            virtualsets.append(CustomDataset(ImageFolder(kaggle_train_path, transform=transform_tensor), args.data_size, transform=transform_kaggle2))

        elif public_dataset == 'rim':
            generatsets.append(CustomDataset(ImageFolder(rim_train_path, transform=transform_tensor), args.data_size, transform=transform_rim2))
            virtualsets.append(CustomDataset(ImageFolder(rim_train_path, transform=transform_tensor), args.data_size, transform=transform_rim2))

        elif public_dataset == 'refuge':
            generatsets.append(CustomDataset(ImageFolder(refuge_train_path, transform=transform_tensor), args.data_size, transform=transform_refuge2))
            virtualsets.append(CustomDataset(ImageFolder(refuge_train_path, transform=transform_tensor), args.data_size, transform=transform_refuge2))

        elif public_dataset == 'cifar':
            # generatsets.append(cifar_trainset)
            # virtualsets.append(cifar_virtualset)
            generatsets.append(cifar_generateset)
            virtualsets.append(cifar_virtualset)

        elif public_dataset == 'self':
            generatsets.append(self_trainset)
            virtualsets.append(self_virtualset)

        elif public_dataset == 'VHL':
            generatsets.append(VHL_generateset)
            virtualsets.append(VHL_virtualset)
    
    
    # unnormalized_train_datasets = [unnormalized_drishti_trainset, unnormalized_kaggle_trainset, unnormalized_rim_trainset, unnormalized_refuge_trainset]
    # train_datasets = [drishti_trainset, kaggle_trainset, rim_trainset, refuge_trainset]
    # test_datasets = [drishti_testset, kaggle_testset, rim_testset, refuge_testset]
    #
    # min_data_len = min(len(drishti_testset), len(kaggle_testset), len(rim_testset), len(refuge_testset))
    # # min_data_len = min(len(kaggle_testset), len(rim_testset), len(refuge_testset))
    #
    # shuffled_idxes = [list(range(0, len(test_datasets[idx]))) for idx in range(len(test_datasets))]
    # for idx in range(len(shuffled_idxes)):
    #     random.shuffle(shuffled_idxes[idx])
    # concated_test_set = [torch.utils.data.Subset(test_datasets[idx], shuffled_idxes[idx][:min_data_len]) for idx in range(len(test_datasets))]
    # concated_test_set = torch.utils.data.ConcatDataset(concated_test_set)
    # print(len(drishti_testset), len(kaggle_testset), len(rim_testset), len(refuge_testset), len(concated_test_set))
    # concated_test_loader = torch.utils.data.DataLoader(concated_test_set, batch_size=args.batch, shuffle=False)

    # return train_datasets, test_datasets, train_loaders, test_loaders, concated_test_loader, unnormalized_train_datasets
    return trainsets, virtualsets, testsets, generatsets

def train(model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)

        loss = loss_fun(output, y)

        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/num_data

def train_vhl(model, optimizer, loss_fun, client_num, device, train_loader, server_images, server_labels, distance_loss, lambda_sim, imgs, server_imgs, ipc, reg_loss):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    align_loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)

        classification_loss = loss_fun(output, y)

        # similarity model update
        # Constrastive
        if reg_loss == 'contrastive':
            client_features = model.embed(x)
            server_features = model.embed(server_images)
            align_loss = distance_loss(client_features, server_features, y, server_labels)
        # MMD
        elif reg_loss == 'mmd':
            for c in range(num_classes):
                client_img_tmp = imgs[c*ipc:(c+1)*ipc]
                server_img_tmp = server_imgs[c*(ipc):(c+1)*(ipc)]
                emb_client = model.embed(client_img_tmp)
                emb_server = model.embed(server_img_tmp)
                align_loss = torch.sum((torch.mean(emb_server, dim=0) - torch.mean(emb_client, dim=0))**2)
        # l2 norm
        elif reg_loss == 'l2norm':
            for c in range(num_classes):
                client_img_tmp = imgs[c*ipc:(c+1)*ipc]
                server_img_tmp = server_imgs[c*(ipc):(c+1)*(ipc)]
                emb_client = model.embed(client_img_tmp)
                emb_server = model.embed(server_img_tmp)
                align_loss = distance_loss(emb_client, emb_server)
        else:
            raise NotImplementedError

        loss = classification_loss + lambda_sim * align_loss

        loss.backward()
        loss_all += loss.item()
        align_loss_all += align_loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/num_data, align_loss_all/len(train_iter)
    

def train_fedprox(args, model, server_model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)

        loss = loss_fun(output, y)

        #########################we implement FedProx Here###########################
        # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
        if step>0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss += args.mu / 2. * w_diff
        #############################################################################

        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/num_data

def test(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []

    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        targets.append(target.detach().cpu().numpy())

        output = model(data)
        
        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()
    
    return test_loss/len(test_loader), correct /len(test_loader.dataset)

################# Key Function ########################
def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models


def get_images(images_all, indices_class, c, n): # get random n images from class c
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return images_all[idx_shuffle]

# def compute_img_mean_std(img_loader):
#     mean = 0.
#     std = 0.
#     nb_samples = 0.
#     for i, (img, label) in enumerate(img_loader):
#         batch_samples = img.size(0)
#         img = img.view(batch_samples, img.size(1), -1)
#         mean += img.mean(2).sum(0)
#         std += img.std(2).sum(0)
#         nb_samples += batch_samples

#     mean /= nb_samples
#     std /= nb_samples
#     print("normMean = {}".format(mean))
#     print("normStd = {}".format(std))
#     return mean, std


# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     print('Device:', device)
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--log', action='store_true', help ='whether to make a log')
#     parser.add_argument('--test', action='store_true', help ='test the pretrained model')
#     parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
#     parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
#     parser.add_argument('--batch', type = int, default= 32, help ='batch size')
#     parser.add_argument('--iters', type = int, default=100, help = 'iterations for communication')
#     parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
#     parser.add_argument('--mode', type = str, default='feddc', help='fedavg | fedprox | fedbn | feddc')
#     parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
#     parser.add_argument('--save_path', type = str, default='./checkpoint/retina', help='path to save the checkpoint')
#     parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
#
#     parser.add_argument('--ipc', type = int, default=10, help = 'images per class')
#     parser.add_argument('--lr_img', type = float, default=1, help = 'learning rate for img')
#     parser.add_argument('--dis_metric', type = str, default='ours', help='matching method')
#     parser.add_argument('--model', type=str, default='ConvNet', help='model')
#     parser.add_argument('--lambda_sim', type = float, default=0.1, help = 'lambda for heterogeneous training')
#     parser.add_argument('--seed', type = int, default=1234, help = 'random seeds')
#
#     parser.add_argument('--ci_iter', type = int, default=200, help = 'client image update epoch')
#     parser.add_argument('--si_iter', type = int, default=2000, help = 'server image update epoch')
#
#     parser.add_argument('--ci_tgap', type = int, default=5, help = 'client image training frequency')
#     parser.add_argument('--si_tgap', type = int, default=5, help = 'server image training frequency')
#     parser.add_argument('--image_update_times', type = int, default=10, help = 'condensed image update times during whole training')
#
#     parser.add_argument('--init', type = str, default='normal', help='initialization method for dc')
#
#     parser.add_argument('--reg_loss', type = str, default='contrastive', help='regularization loss: contrastive|l2norm|mmd')
#
#     args = parser.parse_args()
#     args.device = device
#
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)
#     random.seed(args.seed)
#
#     exp_folder = 'federated_retina'
#
#     args.save_path = os.path.join(args.save_path, exp_folder)
#
#     # log = args.log
#     # if log:
#     #     log_path = os.path.join('../logs/digits/', exp_folder)
#     #     if not os.path.exists(log_path):
#     #         os.makedirs(log_path)
#     #     logfile = open(os.path.join(log_path,'{}.log'.format(args.mode)), 'a')
#     #     logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
#     #     logfile.write('===Setting===\n')
#     #     logfile.write('    lr: {}\n'.format(args.lr))
#     #     logfile.write('    batch: {}\n'.format(args.batch))
#     #     logfile.write('    iters: {}\n'.format(args.iters))
#     #     logfile.write('    wk_iters: {}\n'.format(args.wk_iters))
#
#     if not os.path.exists(args.save_path):
#         os.makedirs(args.save_path)
#     SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))
#     if not os.path.exists(SAVE_PATH):
#         os.makedirs(SAVE_PATH)
#
#     num_classes, channel, im_size, image_batch = 2, 3, (96, 96), 256
#     client_iteration, server_iteration, warmup_iteration = args.ci_iter, args.si_iter, 10000
#     # server_model = DigitModel().to(device)
#     server_model = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model.to(device)
#     loss_fun = nn.CrossEntropyLoss()
#
#     # prepare the data
#     train_datasets, test_datasets, train_loaders, test_loaders, concated_test_loader, unnormalized_train_datasets = prepare_data(args, im_size)
#     # print([len(trainset) for trainset in train_datasets])
#     # print([len(testset) for testset in test_datasets])
#     # sys.exit()
#
#     # name of each client dataset
#     datasets = ['Drishti', 'Kaggle', 'Rim', 'Refuge']
#
#     # federated setting
#     client_num = len(datasets)
#     client_weights = [1/client_num for i in range(client_num)]
#     models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
#
#     # make save dictionary
#     train_loss_save, train_acc_save, val_loss_save, val_acc_save, test_loss_save, test_acc_save, reg_loss_save, global_loss_save, global_acc_save = {}, {}, {}, {}, {}, {}, {}, {}, {}
#     for client_idx in range(client_num):
#         train_loss_save[f'Client{client_idx}'] = []
#         train_acc_save[f'Client{client_idx}'] = []
#         val_loss_save[f'Client{client_idx}'] = []
#         val_acc_save[f'Client{client_idx}'] = []
#         reg_loss_save[f'Client{client_idx}'] = []
#         global_loss_save[f'Client{client_idx}'] = []
#         global_acc_save[f'Client{client_idx}'] = []
#     train_loss_save[f'mean'] = []
#     train_acc_save[f'mean'] = []
#     val_loss_save[f'mean'] = []
#     val_acc_save[f'mean'] = []
#     test_loss_save[f'GLobal Held Out'] = []
#     test_acc_save[f'GLobal Held Out'] = []
#     reg_loss_save[f'mean'] = []
#     global_loss_save[f'mean'] = []
#     global_acc_save[f'mean'] = []
#
#     if args.test:
#         server_model.load_state_dict(torch.load(f'{SAVE_PATH}/server_model.pt'))
#         for client_idx in range(client_num):
#             models[client_idx].load_state_dict(torch.load(f'{SAVE_PATH}/model_{client_idx}.pt'))
#
#         # testing on heldout global test dataset
#         for test_idx, test_loader in enumerate(test_loaders):
#             _, test_acc = test(models[test_idx], test_loader, loss_fun, device)
#             print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
#
#         # testing on heldout global test dataset
#         test_loss, test_acc = test(server_model, concated_test_loader, loss_fun, device)
#         print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format('Global Held Out', test_loss, test_acc))
#
#     else:
#         wandb.init(project='feddc_retina')
#         wandb.config = {'mode': args.mode,
#                         'model': args.model,
#                         'init': args.init,
#                         'lambda': args.lambda_sim}
#         ''' Warm Up: Condense local data before FL'''
#
#         # get initial global and local images
#         if args.init == 'normal':
#             image_syns, label_syns, server_image_syn, server_label_syn = get_initial_normal(train_datasets, im_size, num_classes, client_num, args.ipc)
#         elif args.init == 'noise':
#             server_image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
#             server_label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
#             image_syns, label_syns = [], []
#             image_syns = [copy.deepcopy(server_image_syn).to(args.device) for idx in range(client_num)]
#             label_syns = [copy.deepcopy(server_label_syn).to(args.device) for idx in range(client_num)]
#         else:
#             raise NotImplementedError
#
#
#         ''' Pre-processing clients' data '''
#         # from pre-trained
#         pretrained_img_path = f'./pretrained/retina'
#         if os.path.isfile(f'{pretrained_img_path}/{args.model}_{args.ipc}_{args.init}_client0_iter0.png'):
#             path_tmp = f'{pretrained_img_path}/{args.model}_{args.ipc}_{args.init}'
#             image_syns = GetPretrained(path=path_tmp, means=MEANS, stds=STDS, im_size=im_size, num_classes=num_classes, client_num=client_num, device=args.device, ipc = args.ipc, padding = 2)
#             for i, local_syn_images in enumerate(image_syns):
#                 # for ch in range(channel):
#                 #     local_syn_images[:, ch] = (local_syn_images[:, ch] - MEANS[i][ch]) /STDS[i][ch]
#                 local_syn_images.requires_grad = True
#         # local DM
#         else:
#             for client_idx in range(client_num):
#
#                 # organize the real dataset
#                 images_all = []
#                 labels_all = []
#                 indices_class = [[] for c in range(num_classes)]
#                 images_all = [torch.unsqueeze(train_datasets[client_idx][i][0], dim=0) for i in range(len(train_datasets[client_idx]))]
#                 labels_all = [train_datasets[client_idx][i][1] for i in range(len(train_datasets[client_idx]))]
#                 for i, lab in enumerate(labels_all):
#                     indices_class[lab].append(i)
#                 images_all = torch.cat(images_all, dim=0).to(args.device)
#                 labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
#
#                 # for c in range(num_classes):
#                 #     print('client %d trainset: class c = %d: %d real images'%(client_idx, c, len(indices_class[c])))
#
#                 # images_all__ = []
#                 # labels_all__ = []
#                 # indices_class__ = [[] for c in range(num_classes)]
#                 # images_all__ = [torch.unsqueeze(test_datasets[client_idx][i][0], dim=0) for i in range(len(test_datasets[client_idx]))]
#                 # labels_all__ = [test_datasets[client_idx][i][1] for i in range(len(test_datasets[client_idx]))]
#                 # for i, lab in enumerate(labels_all__):
#                 #     indices_class__[lab].append(i)
#
#                 # for c in range(num_classes):
#                 #     print('client %d testset: class c = %d: %d real images'%(client_idx, c, len(indices_class__[c])))
#
#                 # setup optimizer
#                 optimizer_img = torch.optim.SGD([image_syns[client_idx], ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
#                 optimizer_img.zero_grad()
#
#                 for it in range(warmup_iteration):
#                     loss_avg = 0
#                     # get real images for each class
#                     image_real = [get_images(images_all, indices_class, c, image_batch) for c in range(num_classes)]
#                     if 'BN' in args.model:
#                         loss, image_syns[client_idx] = distribution_matching_bn(image_real, image_syns[client_idx], optimizer_img, channel, num_classes, im_size, args.ipc)
#                     else:
#                         loss, image_syns[client_idx] = distribution_matching(image_real, image_syns[client_idx], optimizer_img, channel, num_classes, im_size, args.ipc)
#                     # report averaged loss
#                     loss_avg += loss
#                     loss_avg /= num_classes
#                     if it%100 == 0:
#                         print('%s Initialization:\t client = %2d, iter = %05d, loss = %.4f' % (get_time(), client_idx, it, loss_avg))
#
#
#
#         ''' start training with condensed images '''
#         local_time = 0
#         global_time = 0
#         start_time = time.time()
#         for a_iter in range(0, args.iters+1):
#
#             # # slow down lr for gradient matching
#             # if a_iter < args.si_tgap*args.image_update_times+1:
#             #     model_lr = args.lr/10
#             # else:
#             #     model_lr = args.lr
#             model_lr = args.lr
#
#             ## Save distilled data for future initialization
#             if a_iter == 0:
#                 # save local distilled data
#                 data_path = f'{pretrained_img_path}'
#                 if not os.path.exists(data_path):
#                     os.makedirs(data_path)
#                 for i, local_syn_images in enumerate(image_syns):
#                     save_name = os.path.join(data_path, f'{args.model}_{args.ipc}_{args.init}_client{i}_iter{a_iter}.png')
#                     image_syn_vis = copy.deepcopy(local_syn_images.detach().cpu())
#                     for ch in range(channel):
#                         image_syn_vis[:, ch] = image_syn_vis[:, ch] * STDS[i][ch] + MEANS[i][ch]
#                     image_syn_vis[image_syn_vis<0] = 0.0
#                     image_syn_vis[image_syn_vis>1] = 1.0
#                     save_image(image_syn_vis, save_name, nrow=args.ipc, normalize=True) # Trying normalize = True/False may get better visual effects.
#
#
#             if ((a_iter+1)%args.si_tgap == 0 or a_iter == 0) and a_iter < args.si_tgap*args.image_update_times+1:
#             # if a_iter==199:
#                 # save global distilled data
#                 data_path = f'{SAVE_PATH}/distilled_data'
#                 if not os.path.exists(data_path):
#                     os.makedirs(data_path)
#                 save_name = os.path.join(data_path, f'{args.model}_{args.ipc}_{args.init}_global_iter{a_iter}.png')
#                 image_syn_vis = copy.deepcopy(server_image_syn.detach().cpu())
#                 mean_global = np.mean(MEANS, axis=0)
#                 std_global = np.mean(STDS, axis=0)
#                 for ch in range(channel):
#                     image_syn_vis[:, ch] = image_syn_vis[:, ch] * std_global[ch] + mean_global[ch]
#                 image_syn_vis[image_syn_vis<0] = 0.0
#                 image_syn_vis[image_syn_vis>1] = 1.0
#                 save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.
#
#             if ((a_iter+1)%args.ci_tgap == 0 or a_iter == 0) and a_iter < args.ci_tgap*args.image_update_times+1:
#                 # save local distilled data
#                 for i, local_syn_images in enumerate(image_syns):
#                     save_name = os.path.join(data_path, f'{args.model}_{args.ipc}_{args.init}_client{i}_iter{a_iter}.png')
#                     image_syn_vis = copy.deepcopy(local_syn_images.detach().cpu())
#                     for ch in range(channel):
#                         image_syn_vis[:, ch] = image_syn_vis[:, ch] * STDS[i][ch] + MEANS[i][ch]
#                     image_syn_vis[image_syn_vis<0] = 0.0
#                     image_syn_vis[image_syn_vis>1] = 1.0
#                     save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.
#
#             if a_iter == args.iters:
#                 break
#
#
#             ## Update local condensed data with DM
#             # if a_iter > 0 and a_iter < 10:
#             # if a_iter > 0 and a_iter%10==0:
#             # if a_iter > 0 and a_iter%10==0 and a_iter<101:
#             if a_iter > 0 and a_iter%args.ci_tgap==0 and a_iter < args.ci_tgap*args.image_update_times+1:
#                 tstart = time.time()
#                 for client_idx in range(client_num):
#                     # organize the real dataset
#                     images_all = []
#                     labels_all = []
#                     indices_class = [[] for c in range(num_classes)]
#                     images_all = [torch.unsqueeze(train_datasets[client_idx][i][0], dim=0) for i in range(len(train_datasets[client_idx]))]
#                     labels_all = [train_datasets[client_idx][i][1] for i in range(len(train_datasets[client_idx]))]
#                     for i, lab in enumerate(labels_all):
#                         indices_class[lab].append(i)
#                     images_all = torch.cat(images_all, dim=0).to(args.device)
#                     labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
#
#                     # setup optimizer
#                     optimizer_img = torch.optim.SGD([image_syns[client_idx], ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
#                     optimizer_img.zero_grad()
#                     # get global condensed images as ref
#                     image_server = [copy.deepcopy(server_image_syn[c*(args.ipc):(c+1)*(args.ipc)].detach()).to(args.device) for c in range(num_classes)]
#                     for it in range(client_iteration):
#                         loss_avg = 0
#                         # get real images for each class
#                         image_real = [get_images(images_all, indices_class, c, image_batch) for c in range(num_classes)]
#                         # loss, image_syns[client_idx], sc_loss = distribution_matching(image_real, image_syns[client_idx], optimizer_img, channel, num_classes, im_size, args.ipc, image_server=image_server)
#                         if 'BN' in args.model:
#                             loss, image_syns[client_idx] = distribution_matching_bn(image_real, image_syns[client_idx], optimizer_img, channel, num_classes, im_size, args.ipc, net=server_model)
#                         else:
#                             loss, image_syns[client_idx] = distribution_matching(image_real, image_syns[client_idx], optimizer_img, channel, num_classes, im_size, args.ipc, net=server_model)
#                         # report averaged loss
#                         loss_avg += loss
#                         loss_avg /= num_classes
#                         # if it == iteration-1:
#                         if (it+1)%100==0:
#                             print('%s Local update:\t client = %2d, Total iter:%05d, iter = %05d, loss = %.4f' % (get_time(), client_idx, a_iter, it+1, loss_avg))
#                 local_time += time.time() - tstart
#                 print(f'Local synthetic images update time per iteration: {time.time() - tstart}')
#
#
#
#             ## Ordinary local training with condensed images
#             tstart = time.time()
#             optimizers = [optim.SGD(params=models[idx].parameters(), lr=model_lr) for idx in range(client_num)]
#             if args.reg_loss == 'contrastive':
#                 distance_loss = Distance_loss(device=args.device)
#             elif args.reg_loss == 'l2norm':
#                 distance_loss = nn.MSELoss()
#             elif args.reg_loss == 'mmd':
#                 distance_loss = None
#             else:
#                 raise NotImplementedError
#
#             # deep copy for any unawared modification
#             image_server_tmp = copy.deepcopy(server_image_syn.detach().to(args.device))
#             label_server_tmp = copy.deepcopy(server_label_syn.detach().to(args.device))
#
#             image_syn_evals = [copy.deepcopy(image_syns[idx].detach()).to(args.device) for idx in range(client_num)]
#             label_syn_evals = [copy.deepcopy(label_syns[idx].detach()).to(args.device) for idx in range(client_num)]
#
#             dst_trains = [TensorDataset(torch.cat([image_syn_evals[idx], image_server_tmp], dim=0), torch.cat([label_syn_evals[idx], label_server_tmp], dim=0)) for idx in range(client_num)]
#             ldr_trains = [torch.utils.data.DataLoader(dst_trains[idx], batch_size=args.batch, shuffle=True, num_workers=0) for idx in range(client_num)]
#
#             for wi in range(args.wk_iters):
#                 print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
#                 # train local model using local condensed images
#                 mean_loss = []
#                 for client_idx in range(client_num):
#                     model, optimizer, ldr_train, image_syn_eval = models[client_idx], optimizers[client_idx], ldr_trains[client_idx], image_syn_evals[client_idx]
#                     _, _, align_loss = train_vhl(model, optimizer, loss_fun, client_num, device, ldr_train, image_server_tmp, label_server_tmp, distance_loss, args.lambda_sim, image_syn_eval, image_server_tmp, args.ipc, args.reg_loss)
#                     # record align_loss
#                     wandb.log({f'Align Loss Client {client_idx}': align_loss})
#                     reg_loss_save[f'Client{client_idx}'].append(align_loss)
#                     mean_loss.append(align_loss)
#                     if client_idx == client_num-1:
#                         reg_loss_save['mean'].append(np.mean(mean_loss))
#             local_time += time.time() - tstart
#             print(f'FL pipeline time per iteration: {time.time() - tstart}')
#
#
#             # Update global condensed data with GM
#             # if a_iter < 10:
#             # if a_iter%10==0:
#             # if a_iter%10==0 and a_iter<101:
#             # if a_iter%5==0 and a_iter<51:
#             if  a_iter%args.si_tgap==0 and a_iter < args.si_tgap*args.image_update_times+1:
#                 tstart = time.time()
#                 # setup optimizer and criterion
#                 optimizer_img = torch.optim.SGD([server_image_syn,], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
#                 optimizer_img.zero_grad()
#                 criterion = nn.CrossEntropyLoss().to(args.device)
#
#                 # set dummy gradient template
#                 gw_reals = [[] for _ in range(num_classes)]
#                 local_datas = [copy.deepcopy(image_syns[idx].detach()).to(args.device) for idx in range(client_num)]
#                 local_labels = [copy.deepcopy(label_syns[idx].detach()).to(args.device) for idx in range(client_num)]
#                 for c in range(num_classes):
#                     for client_idx in range(client_num):
#                         local_data = local_datas[client_idx][c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
#                         local_label = local_labels[client_idx][c*args.ipc:(c+1)*args.ipc]
#                         output_local = server_model(local_data)
#                         loss_local = criterion(output_local, local_label)
#                         if gw_reals[c] == []:
#                             gw_real_ = torch.autograd.grad(loss_local.to(args.device), list(server_model.parameters()))
#                             gw_reals[c] = list((_.detach().clone()*client_weights[client_idx] for _ in gw_real_))
#                         else:
#                             gw_real_ = torch.autograd.grad(loss_local.to(args.device), list(server_model.parameters()))
#                             gw_real_ = list((_.detach().clone()*client_weights[client_idx] for _ in gw_real_))
#                             for i, gw in enumerate(gw_real_):
#                                 gw_reals[c][i] += gw
#                 # client_idx = 1
#                 # local_data = copy.deepcopy(image_syns[client_idx].detach()).to(args.device)
#                 # local_label = copy.deepcopy(label_syns[client_idx].detach()).to(args.device)
#                 # output_local = models[client_idx](local_data)
#                 # loss_local = criterion(output_local, local_label)
#                 # gw_real_ = torch.autograd.grad(loss_local.to(args.device), list(models[client_idx].parameters()))
#                 # gw_real = list((_.detach().clone() for _ in gw_real_))
#
#                 local_time += time.time() - tstart
#
#                 # gradient matching
#                 tstart = time.time()
#                 for it in range(server_iteration):
#                     loss_avg = 0
#                     loss, server_image_syn = gradient_matching(args, server_model, criterion, gw_reals, server_image_syn, optimizer_img, channel, num_classes, im_size, args.ipc)
#                     # report averaged loss
#                     loss_avg += loss
#                     loss_avg /= num_classes
#                     # if it == server_iteration-1:
#                     if (it+1)%100==0:
#                         print('Global update:\t Total iter:%05d, local iter = %05d, loss = %.4f' % (a_iter, it, loss_avg))
#                 global_time += time.time()-tstart
#                 print(f'Global synthetic images update time per iteration: {time.time()-tstart}')
#
#
#             ## Aggregation
#             tstart = time.time()
#             server_model, models = communication(args, server_model, models, client_weights)
#             global_time += time.time() - tstart
#
#
#             ## Report after aggregation
#             mean_loss, mean_acc = [], []
#             for client_idx in range(client_num):
#                 model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
#                 train_loss, train_acc = test(model, train_loader, loss_fun, device)
#                 print(' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx] ,train_loss, train_acc))
#                 train_loss_save[f'Client{client_idx}'].append(train_loss)
#                 train_acc_save[f'Client{client_idx}'].append(train_loss)
#                 mean_loss.append(train_loss)
#                 mean_acc.append(train_acc)
#                 if client_idx == client_num-1:
#                     train_loss_save['mean'].append(np.mean(mean_loss))
#                     train_acc_save['mean'].append(np.mean(mean_acc))
#
#             # testing
#             mean_loss, mean_acc = [], []
#             for test_idx, test_loader in enumerate(test_loaders):
#                 test_loss, test_acc = test(models[test_idx], test_loader, loss_fun, device)
#                 print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasets[test_idx], test_loss, test_acc))
#                 wandb.log({f'Test Loss client {test_idx}': test_loss})
#                 wandb.log({f'Test Acc client {test_idx}': test_acc})
#                 val_loss_save[f'Client{test_idx}'].append(test_loss)
#                 val_acc_save[f'Client{test_idx}'].append(test_acc)
#                 mean_loss.append(test_loss)
#                 mean_acc.append(test_acc)
#                 if test_idx == client_num-1:
#                     val_loss_save['mean'].append(np.mean(mean_loss))
#                     val_acc_save['mean'].append(np.mean(mean_acc))
#
#             # testing on heldout global test dataset
#             test_loss, test_acc = test(server_model, concated_test_loader, loss_fun, device)
#             print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format('GLobal Held Out', test_loss, test_acc))
#             wandb.log({'Test Loss global held-out': test_loss})
#             wandb.log({'Test Acc global held-out': test_acc})
#             test_loss_save['GLobal Held Out'].append(test_loss)
#             test_acc_save['GLobal Held Out'].append(test_acc)
#
#         print(f'Total elapsed time: {time.time()-start_time} secs')
#         wandb.log({'Total elapsed time': (time.time()-start_time)/60})
#         print(f'Total server time: {global_time} secs')
#         print(f'Total client time: {local_time} secs')
#
#
#         # Use globally distilled data for training
#
#         # deep copy for any unawared modification
#         server_image_final = copy.deepcopy(server_image_syn.detach()).to(args.device)
#         server_label_final = copy.deepcopy(server_label_syn.detach()).to(args.device)
#         server_dst_train_final = TensorDataset(server_image_final, server_label_final)
#         server_loader_final = torch.utils.data.DataLoader(server_dst_train_final, batch_size=args.ipc, shuffle=True, num_workers=0)
#
#         # get model and set criterion and optimizer
#         server_model_final = get_network(args.model, channel, num_classes, im_size).to(args.device)
#         loss_fun_final = nn.CrossEntropyLoss()
#         optimizer_final = optim.SGD(params=server_model_final.parameters(), lr=args.lr)
#         for wi in range(args.iters*args.wk_iters):
#             # train local model using local condensed images
#             train(server_model_final, server_loader_final, optimizer_final, loss_fun_final, client_num, device)
#
#         # testing on local datasets
#         mean_loss, mean_acc = [], []
#         for test_idx, test_loader in enumerate(test_loaders):
#             test_loss, test_acc = test(server_model_final, test_loader, loss_fun_final, device)
#             wandb.log({f'Final GLobal Test Acc Client {test_idx}': test_acc})
#             print('Final global model {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasets[test_idx], test_loss, test_acc))
#             global_loss_save[f'Client{test_idx}'].append(test_loss)
#             global_acc_save[f'Client{test_idx}'].append(test_acc)
#             mean_loss.append(test_loss)
#             mean_acc.append(test_acc)
#             if test_idx == client_num-1:
#                 global_loss_save['mean'].append(np.mean(mean_loss))
#                 global_acc_save['mean'].append(np.mean(mean_acc))
#
#
#         # Save checkpoint
#         print(' Saving checkpoints to {}...'.format(SAVE_PATH))
#         for client_idx in range(client_num):
#             torch.save(models[client_idx].state_dict(), f'{SAVE_PATH}/model_{client_idx}.pt')
#         torch.save(server_model.state_dict(), f'{SAVE_PATH}/server_model.pt')
#
#         # Save acc and loss results
#         metrics_pd = pd.DataFrame.from_dict(train_loss_save)
#         metrics_pd.to_csv(os.path.join(SAVE_PATH,f"{args.model}_{args.ipc}_{args.init}_{args.reg_loss}{args.lambda_sim}_train_loss_local{args.wk_iters}_{args.seed}.csv"))
#         metrics_pd = pd.DataFrame.from_dict(train_acc_save)
#         metrics_pd.to_csv(os.path.join(SAVE_PATH,f"{args.model}_{args.ipc}_{args.init}_{args.reg_loss}{args.lambda_sim}_train_acc_local{args.wk_iters}_{args.seed}.csv"))
#         metrics_pd = pd.DataFrame.from_dict(val_loss_save)
#         metrics_pd.to_csv(os.path.join(SAVE_PATH,f"{args.model}_{args.ipc}_{args.init}_{args.reg_loss}{args.lambda_sim}_val_loss_local{args.wk_iters}_{args.seed}.csv"))
#         metrics_pd = pd.DataFrame.from_dict(val_acc_save)
#         metrics_pd.to_csv(os.path.join(SAVE_PATH,f"{args.model}_{args.ipc}_{args.init}_{args.reg_loss}{args.lambda_sim}_val_acc_local{args.wk_iters}_{args.seed}.csv"))
#         metrics_pd = pd.DataFrame.from_dict(test_loss_save)
#         metrics_pd.to_csv(os.path.join(SAVE_PATH,f"{args.model}_{args.ipc}_{args.init}_{args.reg_loss}{args.lambda_sim}_test_loss_local{args.wk_iters}_{args.seed}.csv"))
#         metrics_pd = pd.DataFrame.from_dict(test_acc_save)
#         metrics_pd.to_csv(os.path.join(SAVE_PATH,f"{args.model}_{args.ipc}_{args.init}_{args.reg_loss}{args.lambda_sim}_test_acc_local{args.wk_iters}_{args.seed}.csv"))
#
#         metrics_pd = pd.DataFrame.from_dict(reg_loss_save)
#         metrics_pd.to_csv(os.path.join(SAVE_PATH,f"{args.model}_{args.ipc}_{args.init}_{args.reg_loss}{args.lambda_sim}_local{args.wk_iters}_{args.seed}.csv"))
#
#         metrics_pd = pd.DataFrame.from_dict(global_loss_save)
#         metrics_pd.to_csv(os.path.join(SAVE_PATH,f"{args.model}_{args.ipc}_{args.init}_{args.reg_loss}{args.lambda_sim}_global_loss_local{args.wk_iters}_{args.seed}.csv"))
#         metrics_pd = pd.DataFrame.from_dict(global_acc_save)
#         metrics_pd.to_csv(os.path.join(SAVE_PATH,f"{args.model}_{args.ipc}_{args.init}_{args.reg_loss}{args.lambda_sim}_global_acc_local{args.wk_iters}_{args.seed}.csv"))


