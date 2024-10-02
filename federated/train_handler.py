import random
import time
import warnings
import sys
import argparse
import copy
import numpy as np
import os
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import copy

from loss import DistillationLoss,contrastive_loss, get_features_anchor, Distance_loss

from dalib.adaptation.sfda import mut_info_loss


import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from compute_features import compute_features
import torch.nn as nn
from loss import DistillationLoss,contrastive_loss, get_features_anchor
import umap.umap_ as umap

def fit_umap(tg_models,test_loaders,testsets,device,client_num):
    for tg_model in tg_models:
        tg_model.eval()
    # num_samples = len(testset.images)
    num_features = tg_models[0].head.in_features
    # print(num_samples, num_features)
    test_embeddings = []
    test_targets = []
    test_preds = []
    # test_clients = torch.zeros((0), dtype=torch.float32)
    for c in range(client_num):
        test_embedding = torch.zeros((0, num_features), dtype=torch.float32)
        test_target = torch.zeros((0), dtype=torch.float32)
        test_pred = torch.zeros((0), dtype=torch.float32)
        if c >= len(tg_models):
            model = tg_models[c-len(tg_models)]
        else:
            model = tg_models[c]
        for i, (images_t, target) in enumerate(test_loaders[c]):
            y, embeddings = model(images_t.to(device))
            test_embedding = torch.cat((test_embedding, embeddings.detach().cpu()), 0)
            pred = y.data.max(1)[1]
            # print(pred.shape)
            test_target = torch.cat((test_target, target.cpu()), 0)
            test_pred= torch.cat((test_pred, pred.cpu()), 0)
        test_embeddings.append(test_embedding)
        test_targets.append(test_target)
        test_preds.append(test_pred)
       
    
    # tsne = TSNE(2, verbose=1)
    trans = umap.UMAP(n_neighbors=5, random_state=42).fit(np.concatenate(test_embeddings))
    return trans


def visualize(tg_model,test_loader,testset,ax1,ax2,device,client_num,trans):
    tg_model.eval()
    num_samples = len(testset.images)
    num_features = tg_model.head.in_features
    print(num_samples, num_features)
    test_embeddings = torch.zeros((0, num_features), dtype=torch.float32)
    test_targets = torch.zeros((0), dtype=torch.float32)
    test_preds = torch.zeros((0), dtype=torch.float32)
    for i, (images_t, target) in enumerate(test_loader):
        y, embeddings = tg_model(images_t.to(device))
        test_embeddings = torch.cat((test_embeddings, embeddings.detach().cpu()), 0)
        pred = y.data.max(1)[1]
        # print(pred.shape)
        test_targets = torch.cat((test_targets, target.cpu()), 0)
        test_preds = torch.cat((test_preds, pred.cpu()), 0)
    
    # tsne = TSNE(2, verbose=1)
    tsne_proj = trans.transform(test_embeddings)
    # Plot those points as a scatter plot and label them based on the pred labels
    # co_matrix,test_predictions = compute_confusion_matrix(tg_model, test_loader, device=device)
    # fig, ax = plt.subplots(figsize=(8,8))
    num_categories = tg_model.head.out_features
    cmap = cm.get_cmap('tab20')
    for lab in range(num_categories):
        indices = test_preds==lab
        ax1.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
        indices = test_targets==lab
        ax2.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
    ax1.legend(fontsize='large', markerscale=2)
    ax2.legend(fontsize='large', markerscale=2)
    
    
def visualize_all(tg_models,test_loaders,testsets,ax1,ax2,device,client_num,trans):
    for tg_model in tg_models:
        tg_model.eval()
    # num_samples = len(testset.images)
    num_features = tg_models[0].head.in_features
    # print(num_samples, num_features)
    test_embeddings = []
    test_targets = []
    test_preds = []
    # test_clients = torch.zeros((0), dtype=torch.float32)
    for c in range(client_num):
        test_embedding = torch.zeros((0, num_features), dtype=torch.float32)
        test_target = torch.zeros((0), dtype=torch.float32)
        test_pred = torch.zeros((0), dtype=torch.float32)
        if c >= len(tg_models):
            model = tg_models[c-len(tg_models)]
        else:
            model = tg_models[c]
        for i, (images_t, target) in enumerate(test_loaders[c]):
            y, embeddings = model(images_t.to(device))
            test_embedding = torch.cat((test_embedding, embeddings.detach().cpu()), 0)
            pred = y.data.max(1)[1]
            # print(pred.shape)
            test_target = torch.cat((test_target, target.cpu()), 0)
            test_pred= torch.cat((test_pred, pred.cpu()), 0)
        test_embeddings.append(test_embedding)
        test_targets.append(test_target)
        # print(len(test_pred))
        test_preds.append(test_pred)
    # test_embeddings= np.array(test_embeddings)
    # test_targets= np.array(test_targets)
    # test_preds= np.array(test_preds)
    
    # tsne = TSNE(2, verbose=1)
    # shape = test_embeddings.shape
    # print(shape)
    tsne_proj = trans.transform(np.concatenate(test_embeddings))
    devided = []
    start = 0
    for c in range(client_num):
        num_samples = len(testsets[c].images)
        # print(num_samples)
        devided.append(tsne_proj[start:start+num_samples,])
        # print(len(devided[c]))
        start += num_samples
    # Plot those points as a scatter plot and label them based on the pred labels
    # co_matrix,test_predictions = compute_confusion_matrix(tg_model, test_loader, device=device)
    # fig, ax = plt.subplots(figsize=(8,8))
    num_categories = tg_models[0].head.out_features
    cmap = cm.get_cmap('tab20')
    markers = ["." , "," , "*" , "v" , "^" , "<", ">"]
    for c in range(client_num):
        for lab in range(num_categories):
            indices = test_preds[c]==lab
            # print(len(indices),len(devided[c]))
            ax1.scatter(devided[c][indices,0],devided[c][indices,1], c=np.array(cmap(lab)).reshape(1,4), marker = markers[c],label = lab ,alpha=0.5)
            indices = test_targets[c]==lab
            ax2.scatter(devided[c][indices,0],devided[c][indices,1], c=np.array(cmap(lab)).reshape(1,4), marker = markers[c], label = lab ,alpha=0.5)
    ax1.legend(fontsize='large', markerscale=2,loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.legend(fontsize='large', markerscale=2,loc='center left', bbox_to_anchor=(1, 0.5))
    
    # plt.savefig('/home/s.ayromlou/FedBN/federated/tsne/'+t+'_tsne_map_'+str(client_num)+'_'+str(iteration)+'.png')

def train_uda(trg_loader: DataLoader, src_loader: DataLoader, trg_model, domain_adv, optimizer: SGD, epoch: int, args: argparse.Namespace, device, wandb,client_idx):
    
    # batch_time = AverageMeter('Time', ':5.2f')
    # losses = AverageMeter('Loss', ':6.2f')
    num_data = 0
    correct = 0
    loss_all = 0
 
    trg_model.train()
    
    preds = []
    targets = []
    first_run = True
    iter_num = 0

    if args.fix:
        fixed_model = copy.deepcopy(trg_model)
        for param in fixed_model.parameters():
            param.requires_grad = False
        fixed_model.eval()
    
    for e in range(10*epoch):
        src_iter = iter(src_loader)
        for i, (images_t, target) in enumerate(trg_loader):
            iter_num += 1

            try:
                images_s, labels_s = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                images_s, labels_s = next(src_iter)
            # print('here',images_s.shape,images_t.shape)

            images_s = images_s.to(device)
            labels_s = labels_s.to(device)
            images_t = images_t.to(device)
            target = target.to(device)

            
            y_t, f_t = trg_model(images_t)
            # print('here2')
            if args.fix:
                y_s, f_s = fixed_model(images_s)
            else:
                y_s, f_s = trg_model(images_s)

            # print(y_t.shape, f_t.shape,target.shape)
            loss = 0.
            if args.param_cls_s > 0:
                if args.train_mode == 'original':
                    cls_loss_s = F.cross_entropy(y_t, target)
                elif args.train_mode == 'synthesized':
                    cls_loss_s = F.cross_entropy(y_s, labels_s)
                # print(cls_loss_s)
                if e+1 % 10 == 0:
                    loss += cls_loss_s * args.param_cls_s
                else:
                    loss += cls_loss_s * 0


            if args.uda_type == 'dann' and  args.param_dann > 0:
                dann_loss = domain_adv(f_s, f_t)
                loss += dann_loss * args.param_dann

            if args.uda_type == 'cdan' and  args.param_cdan > 0:
                # print('here')
                cdan_loss = domain_adv(y_s, f_s, y_t, f_t)
                # print(cdan_loss)
                # args.param_cdan = 0
                # if e ==0:
                #     loss += cls_loss_s * 0
                # else:
                loss += cdan_loss * args.param_cdan/50
                # loss += cdan_loss * 0


            if args.param_mi > 0:
                p_t = F.softmax(y_t, dim=1)
                mi_loss = mut_info_loss(p_t)
                loss += mi_loss * args.param_mi

            # losses.update(loss.item(), images_t.size(0))
            loss_all += loss.item()
            num_data += images_t.size(0)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.log:
                metrics = {"Discriminatir_acc" + str(client_idx): domain_adv.domain_discriminator_accuracy,
                           "CE_loss" + str(client_idx): cls_loss_s,
                          "cdan_loss"+ str(client_idx): cdan_loss}
                wandb.log(metrics)


            pred = y_t.data.max(1)[1]

            correct += pred.eq(target.to(device).view(-1)).sum().item()
        
    return loss_all / iter_num, correct / num_data
        


            
            
def train(args, wandb,model, train_loader, optimizer, loss_fun, client_num, device,client_idx,epoch = 1):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    preds = []
    targets = []
    first_run = True
    iter_num = 0
    for e in range(epoch):
        train_iter = iter(train_loader)
        for step in range(len(train_iter)):
            iter_num += 1
            optimizer.zero_grad()
            x, y = next(train_iter)
            num_data += y.size(0)
            x = x.to(device).float()
            y = y.to(device).long()
            output,_ = model(x)

            loss = loss_fun(output, y)
            loss.backward()
            loss_all += loss.item()
            optimizer.step()

            pred = output.data.max(1)[1]
            correct += pred.eq(y.view(-1)).sum().item()
            if step%20 ==0 and args.log:
                metrics = {"CE_loss" + str(client_idx): loss}
                wandb.log(metrics)
    return loss_all / iter_num, correct / num_data


def train_multi_datasets(args, wandb,model, train_loaders, optimizer, loss_fun, client_num, device,client_idx,epoch = 1):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    preds = []
    targets = []
    first_run = True
    iter_num = 0
    oop = 0
    len_batch = [0 for i in range(len(train_loaders))]
    distance_loss = Distance_loss(device=device)
    for e in range(epoch):
        train_iters = []
        for train_loader in train_loaders:
            train_iters.append(iter(train_loader))
        for step in range(len(train_iters[0])):
            iter_num += 1
            optimizer.zero_grad()
            x = None
            y = None
            count = 0
            for train_iter in train_iters:
                if x is None:
                    x, y = next(train_iter)
                    len_batch[count] = len(y)
                else:
                    try:
                        x_temp, y_temp = next(train_iter)

                    except:
                        train_iters[1] = iter(train_loaders[1])
                        x_temp, y_temp = next(train_iters[1])
                    len_batch[count] = len_batch[count - 1] + len(y_temp)
                    x = torch.cat((x, x_temp))
                    y = torch.cat((y, y_temp))
                    # if step==0:
                    # print(x_temp[0])
                    # plt.imshow(np.moveaxis(x_temp[0].numpy(), 0, -1))
                    # plt.savefig('../images/hiii' + str(oop))
                    # oop+=1
                            # print(y)
                count += 1
            num_data += y.size(0)
            x = x.to(device).float()
            y = y.to(device).long()
            output,f = model(x)

            loss = loss_fun(output, y)

            if args.public_dataset == 7:
                align_loss = distance_loss(f[len_batch[0]:len_batch[1]], f[0:len_batch[0]],
                                           y[len_batch[0]:len_batch[1]], y[0:len_batch[0]])
                loss += align_loss
            loss.backward()
            loss_all += loss.item()
            optimizer.step()

            pred = output.data.max(1)[1]
            correct += pred.eq(y.view(-1)).sum().item()
            if step%20 ==0 and args.log:
                metrics = {"CE_loss" + str(client_idx): loss}
                wandb.log(metrics)
    return loss_all / iter_num, correct / num_data


def train_fedprox(args, wandb, server_model, model, train_loader, optimizer, loss_fun, client_num, device,client_idx, epoch = 1):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    first_run = True
    iter_num = 0
    for e in range(epoch):
        train_iter = iter(train_loader)
        for step in range(len(train_iter)):
            iter_num += 1
            optimizer.zero_grad()
            x, y = next(train_iter)
            num_data += y.size(0)
            x = x.to(device).float()
            y = y.to(device).long()
            output,_ = model(x)

            loss = loss_fun(output, y)

            #########################we implement FedProx Here###########################
            # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
            if step > 0:
                w_diff = torch.tensor(0., device=device)
                for w, w_t in zip(server_model.parameters(), model.parameters()):
                    w_diff += torch.pow(torch.norm(w - w_t), 2)
                loss += args.mu / 2. * w_diff
            #############################################################################

            loss.backward()
            loss_all += loss.item()
            optimizer.step()
            if step%20 ==0 and args.log:
                metrics = {"CE_loss" + str(client_idx): loss}
                wandb.log(metrics)
            pred = output.data.max(1)[1]
            correct += pred.eq(y.view(-1)).sum().item()
    return loss_all / iter_num , correct / num_data


def test(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []
    first_run = True
    iter_num = 0
    for data, target in test_loader:
        iter_num += 1
        data = data.to(device).float()
        target = target.to(device).long()
        targets.append(target.detach().cpu().numpy())

        output,_ = model(data)

        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()

    return test_loss / iter_num , correct / len(test_loader.dataset)


################# Key Function ########################
def communication(args, server_model, models, client_weights,client_num,iteration):
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
                    if (args.mode.lower() == 'fedda' and iteration >args.pre_iter and ((iteration < args.pre_merge and args.merge) or not args.merge )) or args.mode.lower() == 'fednorm':
                        print('what')
                        pass
                    else:
                        # print('merging')
                        for client_idx in range(len(client_weights)):
                            models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models


def pgd_attack(model, data, labels, loss_fun, device, maxim, eps=0.05, alpha=0.025, iters=5):
    data = data.to(device)
    # print(data)
    labels = labels.to(device)

    ori_data = data.data

    # print('attack')

    for i in range(iters):
        data.requires_grad = True
        outputs, _ = model(data)
        # print(outputs)
        # print(labels)

        model.zero_grad()
        cost = loss_fun(outputs, labels).to(device)
        cost.backward()

        if maxim:
            adv_data = data + alpha * data.grad.sign()
        else:
            adv_data = data - alpha * data.grad.sign()
        eta = torch.clamp(adv_data - ori_data, min=-eps, max=eps)
        #       data = torch.clamp(ori_data + eta, min=0, max=1).detach_()
        data = ori_data + eta
        data = data.detach_()

        # print(cost)?

    # return data.to(torch.device("cpu"))
    return data.to(torch.device("cpu"))


def train_fedvss(args, wandb, server_model, model, train_loader, optimizer, loss_fun, client_num, device,client_idx,epoch = 1):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    preds = []
    targets = []
    first_run = True
    iter_num = 0
    for e in range(epoch):
        train_iter = iter(train_loader)
        for step in range(len(train_iter)):
            iter_num += 1
            x, y = next(train_iter)
            x_adv_local = pgd_attack(model, x, y, loss_fun, device, True)
            x_adv_global = pgd_attack(server_model, x, y, loss_fun, device, False)
            # x = torch.cat((x, x_adv_local, x_adv_global))
            # y = torch.cat((y, y, y))
            optimizer.zero_grad()
            num_data += y.size(0)
            x = x.to(device).float()
            x_adv_local = x_adv_local.to(device).float()
            x_adv_global = x_adv_global.to(device).float()
            y = y.to(device).long()
            output, _ = model(x)
            output_local, _ = model(x_adv_local)
            output_global, _ = model(x_adv_global)

            loss = loss_fun(output, y) + 0.1 * loss_fun(output_local, y) + loss_fun(output_global, y)
            loss.backward()
            loss_all += loss.item()
            optimizer.step()

            pred = output.data.max(1)[1]
            correct += pred.eq(y.view(-1)).sum().item()
            if step%20 ==0 and args.log:
                metrics = {"CE_loss" + str(client_idx): loss}
                wandb.log(metrics)
    return loss_all / iter_num, correct / num_data