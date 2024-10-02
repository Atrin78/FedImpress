from os import error, replace
import pdb
from typing import Tuple, Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import os
import shutil

task_list = {
    "A": "amazon",
    "D": "dslr",
    "W": "webcam",
    "Ar": "Art",
    "Cl": "Clipart",
    "Pr": "Product",
    "Rw": "Real_World",
    "T": "train",
    "V": "validation"
}


class DataSetIdx(torch.utils.data.Dataset):
    """ pytorch Dataset that return image index too"""
    def __init__(self, dt):
        self.dt = dt

    def __getitem__(self, index):
        data, target = self.dt[index]
        return data, target, index

    def __len__(self):
        return len(self.dt)


class DataSetPath(torch.utils.data.Dataset):
    """ pytorch Dataset that return image index too"""
    def __init__(self, dt):
        self.dt = dt
        self.num_classes = dt.num_classes

    def __getitem__(self, index):
        img, target = self.dt[index]
        path, _ = self.dt.data[index]
        return img, target, path

    def __len__(self):
        return len(self.dt)


class DataSetGen(torch.utils.data.Dataset):
    """ pytorch Dataset that return image index too"""
    def __init__(self, data_list_file: str):
        self.data = self.parse_data_file(data_list_file)

    def __getitem__(self, index):
        path, target = self.data[index]
        img = torch.load(path)
        return img, target

    def __len__(self):
        return len(self.data)

    def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        """Parse file to data list

        Parameters:
            - **file_name** (str): The path of data file
            - **return** (list): List of (image path, class_index) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                path, target = line.split()
                target = int(target)
                data_list.append((path, target))
        return data_list


class DataSetSample(torch.utils.data.Dataset):
    """ pytorch Dataset that return image index too"""
    def __init__(self, data_list_file: str):
        self.data = self.parse_data_file(data_list_file)
        self.loader = default_loader
        self.num_classes = 31

    def __getitem__(self, index):
        path, target = self.data[index]
        img = self.loader(path)
        return img, target

    def __len__(self):
        return len(self.data)
    
    def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        """Parse file to data list

        Parameters:
            - **file_name** (str): The path of data file
            - **return** (list): List of (image path, class_index) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                path, target = line.split()
                target = int(target)
                data_list.append((path, target))
        return data_list


class DataSetTransform(torch.utils.data.Dataset):
    def __init__(self, dt, transform):
        self.dt = dt
        self.data = dt.data
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.dt[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.dt)


class LabelOptimizer():

    def __init__(self, N, K, lambd, device):
        self.N = N
        self.K = K
        self.lambd = lambd
        self.device = device
        self.P = torch.zeros(N, K).to(device)
        self.Q = torch.zeros(N, K).to(device)
        self.Labels = torch.zeros(N).to(device)
        self.r = 1.
        self.c = 1. * N / K

    def update_P(self, p_t, index):
        # p_batch = p_t / self.N
        self.P[index, :] = p_t

    def update_Labels(self):
        # solve label assignment via sinkhorn-knopp
        self.P = self.P ** self.lambd
        v = (torch.ones(self.K, 1) / self.K).to(self.device)
        err = 1.
        cnt = 0
        while err > 0.1:
            u = self.r / (self.P @ v)
            new_v = self.c / (self.P.T @ u)
            err = torch.sum(torch.abs(new_v / v - 1))
            v = new_v
            cnt += 1
        print(f'error: {err}, step: {cnt}')
        self.Q = u * self.P * v.squeeze()
        # Q = torch.diag(u.squeeze()) @ self.P @ torch.diag(v.squeeze())
        self.Labels = self.Q.argmax(dim=1)


def entropy_loss(p_t):
    # return - (p_t * torch.log(p_t + 1e-5)).sum() / p_t.size(0)
    return (- (p_t * torch.log(p_t + 1e-5)).sum(dim=1)).mean()


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VirtualAdversarialLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VirtualAdversarialLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x)[0], dim=1)

        # prepare random unit tensor
        d = torch.randn(x.shape).to(x.device)
        d = _l2_normalize(d)

        # calc adversarial direction
        for _ in range(self.ip):
            d.requires_grad_()
            pred_hat, _ = model(x + self.xi * d)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
            adv_distance.backward()
            d = _l2_normalize(d.grad)
            model.zero_grad()
    
        # calc VAT loss
        r_adv = d * self.eps
        pred_hat, _ = model(x + r_adv)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        loss = F.kl_div(logp_hat, pred, reduction='batchmean')

        return loss


def cross_entropy_ls(pred, label, alpha=0.1):
    ce_loss = F.cross_entropy(pred, label)
    kl_loss = - torch.mean(F.log_softmax(pred, dim=1))
    return (1 - alpha) * ce_loss + alpha * kl_loss


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort


def weight_reg_loss(src_model, trg_model):
    weight_loss = 0
    for src_param, trg_param in zip(src_model.parameters(), trg_model.parameters()):
        weight_loss += ((src_param - trg_param) ** 2).sum()
    return weight_loss


def diversity_loss(p_t):
    p_mean = p_t.mean(dim=0)
    return (p_mean * torch.log(p_mean + 1e-5)).sum()


def mut_info_loss(p_s):
    ent = (- (p_s * torch.log(p_s + 1e-5)).sum(dim=1)).mean()
    p_mean = p_s.mean(dim=0)
    div = (p_mean * torch.log(p_mean + 1e-5)).sum() + torch.log(torch.tensor(1. * p_s.size(1)))
    return ent + div


def dist_loss(a, b, mode='l2'):
    if mode == 'l2':
        return F.mse_loss(a, b, reduction='sum') / a.size(0)
    if mode == 'l1':
        return F.l1_loss(a, b, reduction='mean')
    elif mode == 'cosine':
        return (1 - F.cosine_similarity(a.reshape(a.size(0), -1), b.reshape(b.size(0), -1))).mean()


def save_images(src_model, images_s, images_t, labels_t, device):
    # predict labels
    y_s, f_s = src_model(images_s.detach())
    p_s = F.softmax(y_s.detach(), dim=1)
    p_s_max, pl_s = p_s.max(dim=1)
    # pse_labels_s = p_s.argmax(dim=1)
    
    labels_t = labels_t.to(device)
    correct = pl_s.eq(labels_t).sum()
    print(f'correct: {correct}')

    y_t, f_t = src_model(images_t)
    p_t = F.softmax(y_t, dim=1)
    p_t_max, pl_t = p_t.max(dim=1)

    correct_p = pl_t.eq(labels_t).sum()
    print(f'correct_p: {correct_p}')

    dirs1 = 'output'
    if not os.path.exists(dirs1):
        os.makedirs(dirs1)
    else:
        shutil.rmtree(dirs1)
        os.makedirs(dirs1)

    # save imgs
    for i in range(images_t.size(0)):
        pl_s1 = pl_s[i]
        p_s1 = p_s_max[i]
        pl_t1 = pl_t[i]
        p_t1 = p_t_max[i]
        label = labels_t[i]

        img = images_t[i]
        img -= img.min()
        img /= img.max()
        im = transforms.ToPILImage()(img)
        im.save(f'output/{i}_trg_label{label}_pselab{pl_t1}_prob{p_t1:.2f}.jpg', "JPEG")

        img = images_s[i].detach()
        img -= img.min()
        img /= img.max()
        im = transforms.ToPILImage()(img)
        im.save(f'output/{i}_src_pselab{pl_s1}_prob{p_s1:.2f}.jpg', "JPEG")


def save_src_imgs_repeat(images, labels, paths, gen_folder, data_list_file, repeat_times, aug_transform, args):

    for img, label, path in zip(images, labels, paths):
        trg_task = task_list[args.target]
        path = path.replace('data/', gen_folder)
        path = path.replace('.jpg', '.pt')
        path = path.replace('/'+trg_task+'/', '/'+args.source+'2'+args.target+'/')
        
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        torch.save(img.clone(), path)

        data_list = path + ' {}'.format(label.item()) + '\n'
        with open(data_list_file, 'a+') as f:
            f.write(data_list)

        n_repeat = repeat_times[label.item()]
        if n_repeat > 0:
            for i in range(n_repeat):
                img_aug = aug_transform(img)
                path_aug = path.replace('.pt', f'_{i}.pt')
                data_list_aug = path_aug + ' {}'.format(label.item()) + '\n'
                torch.save(img_aug.clone(), path_aug)
                with open(data_list_file, 'a+') as f:
                    f.write(data_list_aug)


def save_src_imgs(images, labels, paths, gen_folder, data_list_file, args):

    for img, label, path in zip(images, labels, paths):
        trg_task = task_list[args.target]
        path = path.replace('data/', gen_folder)
        path = path.replace('.jpg', '.pt')
        path = path.replace('/'+trg_task+'/', '/'+args.source+'2'+args.target+'/')
        
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        torch.save(img.clone(), path)

        data_list = path + ' {}'.format(label.item()) + '\n'
        with open(data_list_file, 'a+') as f:
            f.write(data_list)


class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class MyDistributedDataParallel(nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def mixup_data(x, y, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_ce(pred, y_a, y_b, lam):
    return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        if isinstance(module, nn.BatchNorm2d):
            nch = input[0].shape[1]
            mean = input[0].mean([0, 2, 3])
            var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        elif isinstance(module, nn.BatchNorm1d):
            mean = input[0].mean(0)
            var = input[0].permute(1, 0).contiguous().var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


def get_loss_r_feature_layers(model, args):
    # Create hooks for feature statistics
    loss_r_feature_layers = []
    if args.bn_type == 'last':
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module))
    elif args.bn_type == 'all':
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module))
    return loss_r_feature_layers


def moment_match_loss(f_s, f_t):
    mean_s = f_s.mean(0)
    var_s = f_s.var(0, unbiased=False)
    mean_t = f_t.mean(0)
    var_t = f_t.var(0, unbiased=False)
    return torch.norm(mean_s - mean_t, 2) + torch.norm(var_s - var_t, 2)


def nuclear_norm_loss(p_t):
    return - torch.norm(p_t, 'nuc') / p_t.size(0)
