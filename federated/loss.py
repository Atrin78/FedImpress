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

from dalib.adaptation.sfda import mut_info_loss


import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from compute_features import compute_features
import torch.nn as nn



class DistillationLoss:
    def __init__(self):
        self.student_loss = nn.CrossEntropyLoss()
        self.distillation_loss = nn.KLDivLoss()
        self.temperature = 1
        self.alpha = 0.0

    def __call__(self, student_logits, teacher_logits):
        distillation_loss = self.distillation_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                                   F.softmax(teacher_logits / self.temperature, dim=1))

        # loss = (1 - self.alpha) * student_target_loss + self.alpha * distillation_loss
        return distillation_loss
    
def get_features_anchor(model, anchore_loader, device):
    model.eval()
    features = []
    targets = []
    preds = []
    first_run = True
    t_features = torch.zeros((model.fc.out_features, model.fc.in_features)).to(device)
    num_each = torch.zeros((model.fc.out_features))
    for data, target in anchore_loader:
        data = data.to(device).float()
        target = target.to(device).long()

        y_t, f_t = model(data)
        
        pred = y_t.data.max(1)[1]


        for i in range(len(t_features)):
            indices = torch.where(pred==i)[0]
            if len(indices) != 0:
                num_each[i] += len(indices)
                t_features[i] += torch.sum(
                    f_t[indices],dim=0)
    
    for i in range(len(t_features)):
        if num_each[i] != 0:
            t_features[i] = t_features[i]/num_each[i] 
    t_centroids = t_features.detach()
    t_centroids_norm = t_centroids / (t_centroids.norm(dim=1)[:, None]+1e-10)
    
    return t_centroids_norm

# def contrastive_loss(anchor_data_loader,model,inputs,targets,device):
def contrastive_loss(t_centroids,s_centroids,f_t,outputs,f_s,outputs_s,model,device,exist):
    ro = 0.99
    temprature = 0.1
    t_features = torch.zeros((model.head.out_features, model.head.in_features)).to(device)
    
    # print(outputs)
    for i in range(len(t_features)):
        if exist[i]:
            indices = torch.where(outputs==i)[0]
            if len(indices) != 0:
                t_features[i] = torch.sum(f_t[indices],dim=0)/len(indices)
    # print(s_features[0])

    t_centroids = (1-ro)*t_centroids.detach() + ro*t_features
    # print(s_centroids)
    
    s_features = torch.zeros((model.head.out_features, model.head.in_features)).to(device)
    # print(outputs)
    for i in range(len(s_features)):
        if exist[i]:
            indices = torch.where(outputs_s==i)[0]
            if len(indices) != 0:
                s_features[i] = torch.sum(f_s[indices],dim=0)/len(indices)
    # print(s_features[0])

    s_centroids = (1-ro)*s_centroids.detach() + ro*s_features

    t_centroids_norm = t_centroids / (t_centroids.norm(dim=1)[:, None]+1e-10)
    s_centroids_norm = s_centroids / (s_centroids.norm(dim=1)[:, None]+1e-10)
    res = torch.exp(torch.mm(t_centroids_norm, s_centroids_norm.transpose(0,1))/temprature)

    loss4 = -1* torch.sum(torch.log(torch.diagonal(res,0)))+torch.sum(torch.log(torch.sum(res,dim = 0)))
    
    # Loss = nn.MSELoss()
    # loss4 = 100*Loss(t_centroids_norm,s_centroids_norm)
    
    return loss4,t_centroids,s_centroids



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, contrast_mode='all',
                base_temperature=0.07, device=None):
        super(SupConLoss, self).__init__()
        # self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, temperature=0.07, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), temperature)
        # logging.info(f"In SupCon, anchor_dot_contrast.shape: {anchor_dot_contrast.shape}, anchor_dot_contrast: {anchor_dot_contrast}")
        # logging.info(f"In SupCon, anchor_dot_contrast.shape: {anchor_dot_contrast.shape}, anchor_dot_contrast: {anchor_dot_contrast.mean()}")
        # logging.info(f"In SupCon, anchor_dot_contrast.device: {anchor_dot_contrast.device}, self.device: {self.device}")


        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # logging.info(f"In SupCon, exp_logits.shape: {exp_logits.shape}, exp_logits: {exp_logits.mean()}")
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # if torch.any(torch.isnan(log_prob)):
        #     log_prob[torch.isnan(log_prob)] = 0.0
        # logging.info(f"In SupCon, log_prob.shape: {log_prob.shape}, log_prob: {log_prob.mean()}")

        mask_sum = mask.sum(1)
        mask_sum[mask_sum == 0] += 1

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = - (temperature / self.base_temperature) * mean_log_prob_pos
        # loss[torch.isnan(loss)] = 0.0
        if torch.any(torch.isnan(loss)):
            # loss[torch.isnan(loss)] = 0.0
            # logging.info(f"In SupCon, features.shape: {features.shape}, loss: {loss}")
            raise RuntimeError
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class Distance_loss(nn.Module):
    def __init__(self, distance="SupCon", device=None):
        super(Distance_loss, self).__init__()
        self.distance = distance
        self.device = device
        if self.distance == "SupCon":
            self.supcon_loss = SupConLoss(contrast_mode='all', base_temperature=0.07, device=self.device)
        else:
            self.supcon_loss = None


    def forward(self, x1, x2, label1=None, label2=None):
        if self.distance == "L2_norm":
            loss = self.L2_norm(x1, x2)
        elif self.distance == "cosine":
            loss = self.cosine(x1, x2)
        elif self.distance == "SupCon":
            loss = self.supcon(x1, x2, label1, label2)
        else:
            raise NotImplementedError
        return loss


    def L2_norm(self, x1, x2):
        return (x1 - x2).norm(p=2)

    def cosine(self, x1, x2):
        cos = F.cosine_similarity(x1, x2, dim=-1)
        loss = 1 - cos.mean()
        return loss

    def supcon(self, feature1, feature2, label1, label2):

        all_features = torch.cat([feature1, feature2], dim=0)

        all_features = F.normalize(all_features, dim=1)
        all_features = all_features.unsqueeze(1)

        align_cls_loss = self.supcon_loss(
            features=all_features,
            labels=torch.cat([label1, label2], dim=0),
            temperature=0.07, mask=None)
        return align_cls_loss