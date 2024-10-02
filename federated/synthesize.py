import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import gc
import torch
from typing import Tuple, Optional, List, Dict
from torch import nn, optim
import time
import copy
from nets.models import DigitModel
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from utils import data_utils
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.functional as func
from torch.optim import SGD
import torchvision.utils as vutils
import matplotlib.image as image
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from PIL import Image
import gc
import numpy as np


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()






def clip(image_tensor,mean,std, dim = 1,use_fp16=False):
    '''
    adjust the input based on mean and variance
    '''
    for c in range(dim):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor

def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.2 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def labels_to_one_hot(labels, num_class, device):
    # convert labels to one-hot
    labels_one_hot = torch.FloatTensor(labels.shape[0], num_class).to(device)
    labels_one_hot.zero_()
    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return labels_one_hot


def src_img_synth_ce(gen_loader, src_model, args , device, mode, save_dir,a_iter, class_num, wandb, class_count):

    src_model.eval()

    gen_dataset = None
    gen_labels = None
    original_dataset = None
    original_labels = None

    for batch_idx, (images_t, labels_t) in enumerate(gen_loader):

        if original_labels is None:
            original_dataset = images_t
            original_labels = labels_t
        else:
            original_dataset = torch.cat((original_dataset, images_t))
            original_labels = torch.cat((original_labels, labels_t))

        images_t = images_t.to(device)
        # get pseudo labels
        y_t, _ = src_model(images_t)
        plabel_t = y_t.argmax(dim=1)

        # init src img
        images_s = images_t.clone()
        images_s.requires_grad_()
        optimizer_s = SGD([images_s], args.lr_img, momentum=args.momentum_img)

        for iter_i in range(args.iters_img):
            y_s, _ = src_model(images_s)
            loss = func.cross_entropy(y_s, plabel_t)

            optimizer_s.zero_grad()
            loss.backward()
            optimizer_s.step()

            print(loss)

        # save src imgs
        if gen_dataset is None:
            gen_dataset = images_s.detach_().cpu()
            gen_labels = plabel_t
        else:
            gen_dataset = torch.cat((gen_dataset, images_s.detach_().cpu()))
            gen_labels = torch.cat((gen_labels, plabel_t))

    return gen_dataset, gen_labels, original_dataset ,original_labels

def src_img_synth_admm(gen_loader, src_model, args , device, mode, save_dir,a_iter, class_num, wandb, class_count):

    src_model.eval()
    LAMB = torch.zeros_like(src_model.head.weight.data).to(device)
    gen_dataset = None
    gen_labels = None
    original_dataset = None
    original_labels = None
    if args.add_bn_normalization:
        loss_r_feature_layers = []
        for module in src_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    # metrics = {"Train_ACC_" + str(client_idx): train_acc,
    #                        "Train_Loss_" + str(client_idx): train_loss}
    #             if args.report_f1:
    #                 metrics["Train_F1_" + str(client_idx)] =train_f1
    #             wandb.log(metrics)
    for batch_idx, (images_s, labels_real) in enumerate(gen_loader):
        print(batch_idx,len(images_s))
        # if batch_idx == 10:
        #     break
        images_s = images_s.to(device)
        y_s,_ = src_model(images_s)
        labels_s = y_s.argmax(dim=1)
        if gen_dataset == None:
            gen_dataset = images_s
            if args.synthesize_label == 'cond':
                gen_labels = torch.tensor(np.random.choice(2, len(labels_real), p=class_count/sum(class_count)))
            elif args.synthesize_label == 'pred' or mode == 'test':
                gen_labels = labels_s
            else:
                print('hereee')
                gen_labels = labels_real
            original_dataset = images_s
            original_labels = labels_real
        else:
            gen_dataset = torch.cat((gen_dataset, images_s), 0)
            if args.synthesize_label == 'cond':
                lab = torch.tensor(np.random.choice(2, len(labels_real), p=class_count / sum(class_count)))
                gen_labels = torch.cat((gen_labels, lab), 0)
            elif args.synthesize_label == 'pred' or mode == 'test':
                gen_labels = torch.cat((gen_labels, labels_s), 0)
            else:
                gen_labels = torch.cat((gen_labels, labels_real), 0)
            original_dataset = torch.cat((original_dataset, images_s), 0)
            original_labels = torch.cat((original_labels, labels_real), 0)

    if args.noise_init:
        print('here we are rand')
        gen_dataset = torch.from_numpy(np.random.normal(0, 1
                                                        , (args.data_size, 3, 256, 256))).type(torch.FloatTensor)
        gen_labels = torch.randint(low=0, high=2, size=(args.data_size,))
        # print(gen_dataset[0])
        # gen_dataset = torch.from_numpy(np.random.normal(0, 1
        #                                                 , (args.data_size, 3, 256, 256))).type(torch.FloatTensor)
        # gen_labels = torch.randint(low=0, high=2, size=(args.data_size,))
        # print(gen_dataset[0])
        # print(gen_labels[0:20])
        original_labels = gen_labels.clone()
        original_dataset = gen_dataset.clone()

    for i in range(args.iters_admm):
        

        print(f'admm iter: {i}/{args.iters_admm}')

        # step1: update imgs
        for batch_idx, (images_s, labels_s) in enumerate(gen_loader):
            # if batch_idx == 10:
            #     break

            gc.collect()

    #        images_s = images_s.to(device)
    #        labels_s = labels_s.to(device)
            images_s = gen_dataset[batch_idx*args.batch:(batch_idx+1)*args.batch].clone().detach().to(device)
            labels_s = gen_labels[batch_idx*args.batch:(batch_idx+1)*args.batch].clone().detach().to(device)

            # convert labels to one-hot
            plabel_onehot = labels_to_one_hot(labels_s, class_num, device)

            # init src img
            images_s.requires_grad_()
            optimizer_s = SGD([images_s], args.lr_img, momentum=args.momentum_img)
            first_run = True
            
            for iter_i in range(args.iters_img):
                y_s, f_s = src_model(images_s)
                loss = func.cross_entropy(y_s, labels_s)
                p_s = func.softmax(y_s, dim=1)
                grad_matrix = (p_s - plabel_onehot).t() @ f_s / p_s.size(0)
                new_matrix = grad_matrix + args.param_gamma * src_model.head.weight.data
                grad_loss = torch.norm(new_matrix, p='fro') ** 2
                loss += grad_loss * args.param_admm_rho / 2
                loss += torch.trace(LAMB.t() @ new_matrix)
#                 if args.add_bn_normalization:
#                     rescale = [10] + [1. for _ in range(len(loss_r_feature_layers)-1)]
#                     # if iteration_loc == 0:
#                     #     print("rescale",rescale)
#                     loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

#                     loss += 0.01 * loss_r_feature


                if batch_idx==0:
                    metrics = {"Loss_Genration"+str(i): loss}
                    wandb.log(metrics)
                first_run = False

                optimizer_s.zero_grad()
                loss.backward()
                optimizer_s.step()

                print(loss)

                # images_s.clamp(0.0, 1.0)
                gc.collect()


            # update src imgs
            gen_dataset[batch_idx*args.batch:(batch_idx+1)*args.batch] = images_s
       #     for img, path in zip(images_s.detach_().cpu(), paths):
       #         torch.save(img.clone(), path)

        # step2: update LAMB
        grad_matrix = torch.zeros_like(LAMB).to(device)
        for batch_idx, (images_s, labels_s) in enumerate(gen_loader):
            # if batch_idx == 10:
            #     break
       #     images_s = images_s.to(device)
       #     labels_s = labels_s.to(device)
            images_s = gen_dataset[batch_idx*args.batch:(batch_idx+1)*args.batch].clone().detach().to(device)
            labels_s = gen_labels[batch_idx*args.batch:(batch_idx+1)*args.batch].clone().detach().to(device)

            # convert labels to one-hot
            plabel_onehot = labels_to_one_hot(labels_s, class_num, device)

            y_s, f_s = src_model(images_s)
            p_s = func.softmax(y_s, dim=1)
            grad_matrix += (p_s - plabel_onehot).t() @ f_s

        new_matrix = grad_matrix / len(gen_dataset) + args.param_gamma * src_model.head.weight.data
        LAMB += new_matrix * args.param_admm_rho

        gc.collect()
        
    if args.add_bn_normalization:
        for hook in loss_r_feature_layers:
            hook.close()


    if (a_iter-1) % args.save_every == 0:
        print("saving image dir to", save_dir)
        vutils.save_image(torch.cat((original_dataset[0:20],gen_dataset[0:20]),0), save_dir ,
                          normalize=True, scale_each=True, nrow=int(10))
        # plt.style.use('dark_background')
        # # fig = plt.figure()
        # # ax = fig.add_subplot()
        # image = plt.imread(save_dir)
        # ax.imshow(image)
        # ax.axis('off')
        # fig.set_size_inches(10 * 5, 10*10 )
        # plt.title("ori_labels= "+str(original_labels[0:20])+"\n gen_labels="+str(gen_labels[0:20]), fontweight="bold")
        # plt.savefig(save_dir)
        # plt.close()

    return gen_dataset, gen_labels, original_dataset ,original_labels
