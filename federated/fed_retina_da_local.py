"""
federated learning with different aggregation strategy on benchmark exp.
"""
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
from torch import nn, optim
import time
import copy
from nets.models import DigitModel, AlexNet
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from utils import data_utils
from utils.data_utils import OfficeDataset, CustomDataset
import wandb
import os
from torch.utils.data import TensorDataset
from torchmetrics.classification import BinaryF1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
import random
#import seaborn


### Domain adaptation modules import
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.dann import DomainAdversarialLoss
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss
from train_handler import train_uda, train, train_fedprox, test, communication, visualize, visualize_all,fit_umap, train_multi_datasets, train_fedvss
from synthesize import src_img_synth_admm, src_img_synth_ce
from digit_net import ImageClassifier
from prepare_data import prepare_office
from feddc_retina_nonntk import prepare_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--log', action='store_true', help='whether to make a log')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--percent', type=float, default=0.1, help='percentage of dataset to train')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--iters', type=int, default=100, help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='fedbn', help='fedavg | fedprox | fedbn | fedda | fedvss')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type=str, default='checkpoint2/digits', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')
    parser.add_argument('--project_name', type=str, default='fed_digit_5', help='name of wandb project')

    parser.add_argument('--cuda_num', type=int, default=0, help='cuda num')
    parser.add_argument('--model_arch', type=str, default='digitnet', help='trainining model architecture digitnet|convnet|resnet|alexnet')

    parser.add_argument('--train_mode', type=str, default= None , help='original | synthesized')
    parser.add_argument('--uda_type', default='cdan')
    parser.add_argument('--param_cdan', default=1, type=float)
    parser.add_argument('--param_dann', default=0., type=float)
    parser.add_argument('--param_cls_s', default=1., type=float)
    parser.add_argument('--param_mi', default=0., type=float)
    
    
    parser.add_argument('--synthesize_mode', type=str, default= None , help='local | global')
    parser.add_argument('--synthesize_label', type=str, default= 'cond' , help='real | pred | cond')
    parser.add_argument('--synth_method', type=str, default='admm', help='admm | ce')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--momentum_img', default=0.9, type=float, metavar='M',
                        help='momentum of img optimizer')
    parser.add_argument('--iters_img', default=10, type=int, metavar='N',
                        help='number of total inner epochs to run')
    parser.add_argument('--param_gamma', default=0.01, type=float)
    parser.add_argument('--param_admm_rho', default=0.01, type=float)
    parser.add_argument('--iters_admm', default=3, type=int)
    parser.add_argument('--lr_img', default=100., type=float)
                                   
    
    parser.add_argument('--pre_iter', default= 20 , type=int) 
    parser.add_argument('--pre_merge', default= None , type=int) 
    parser.add_argument('--save_every', default= 10 , type=int)

    parser.add_argument('--client_num', default= 2 , type=int)
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--add_bn_normalization', action='store_true', help='batch norm loss')
    parser.add_argument('--public_dataset', default= 0 , type=int)
    parser.add_argument('--data_size', default=160, type=int)
    parser.add_argument('--runid', default= None , type=str)
    parser.add_argument('--merge', action='store_true', help='merge training for local from servers')
    parser.add_argument('--noise_init', action='store_true', help='synthesize with noise')
    parser.add_argument('--synthesize_test', action='store_true', help='make a virtual test data and report according')
    parser.add_argument('--fix', action='store_true', help='fixes embeddings for virtual dataset')

     
    
    args = parser.parse_args()
                                   

    device = torch.device('cuda:' + str(args.cuda_num) if torch.cuda.is_available() else 'cpu')
    # seed = 2
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    lr_factor = 0.3  # Learning rate decrease factor
    lr_patience = 5
    lr_threshold = 0.0001
    trainset_num_classes = 2

    print('Device:', device)

    Best_Global_model = None
    Best_local_models = None

    if args.pre_merge == None:
        if args.merge:
            args.pre_merge = args.pre_iter
        else:
            args.pre_merge = args.iters

    NAME = args.mode + '_'
    NAME = NAME + 'over_' + str(args.client_num) 
    if args.public_dataset > 0: 
        NAME = NAME + '_pulic_data_' + str(args.public_dataset) 
    if args.train_mode != None:
        NAME = NAME + '_'+ args.train_mode
    if args.synthesize_mode != None:
        NAME = NAME + '_'+ args.synthesize_mode
    if args.merge:
        NAME = NAME + '_merged'
    NAME = NAME + '_wk_iters_' + str(args.wk_iters)
    NAME = NAME +'_param_cdan' + str(args.param_cdan)
        
    NAME = NAME +'_pre_merge_' + str(args.pre_merge)
    NAME = NAME +'_synthesize_label_' + str(args.synthesize_label)
    
    if args.synthesize_test:
        NAME = NAME + '_synthesize_test'
    NAME = NAME + '_model_arch_'+args.model_arch
                                                                      

    print(NAME)
    
    log = args.log
    if log:
        # log_path = args.save_path + SAVE_PATH + '_log'
        os.environ["WANDB_API_KEY"] = 'f87c7a64e4a4c89c4f1afc42620ac211ceb0f926'
        if args.runid != None and args.resume:
            wandb.init(project=args.project_name, entity="sanaayr", id=args.runid , resume="must" ,config=args)
        else:
            wandb.init(project=args.project_name, entity="sanaayr", config=args)
            wandb.run.name = NAME
        wandb.run.save()

                                   
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    FIG_SAVE_PATH = os.path.join(args.save_path, '{}'.format('FIG'))
    Synth_SAVE_PATH = os.path.join(args.save_path, '{}'.format('Synth'))
    if not os.path.exists(FIG_SAVE_PATH):
        os.makedirs(FIG_SAVE_PATH)
    if not os.path.exists(Synth_SAVE_PATH):
        os.makedirs(Synth_SAVE_PATH)
    FIG_SAVE_PATH = os.path.join(FIG_SAVE_PATH, '{}'.format(NAME))
    Synth_SAVE_PATH = os.path.join(Synth_SAVE_PATH, '{}'.format(NAME))
    SAVE_PATH = os.path.join(args.save_path, '{}'.format(NAME))



    # server_model = ImageClassifier(args.model_arch,31, 512).to(device)
    # server_model = DigitModel2(trainset_num_classes).to(device)
    # server_model = DigitModel(trainset_num_classes).to(device)
    server_model = AlexNet(trainset_num_classes).to(device)
    print(server_model)
    loss_fun = nn.CrossEntropyLoss()

    #seaborn.set()
    #seaborn.set_style("white")
    #loss_list_5 = list(np.load('medical_loss_5.npy'))
    #loss_list_10 = list(np.load('medical_loss_10.npy'))
    ## loss_list.append(avg_loss)
    ## print(loss_list)
    #plt.plot(range(1, len(loss_list_5)+1), loss_list_5, label='5 Local Epochs')
    #plt.plot(range(1, len(loss_list_10) + 1), loss_list_10, label='10 Local Epochs')
    #plt.legend(loc='upper right')
    #plt.ylabel('Cross-entropy Loss')
    #plt.xlabel('Communication Round')
    #plt.savefig('loss_curve')
    #exit()
    ## np.save('medical_loss_' + str(args.wk_iters), np.array(loss_list))
    ## exit()


    # name of each client dataset
    datasets = ['drishti', 'kaggle', 'rim', 'refuge', 'cifar', 'self','VHL']
    public_dataset = None
    if args.public_dataset > 0 and args.synthesize_mode == 'global':
        public_dataset = datasets[args.public_dataset - 1]
        datasets.pop(args.public_dataset - 1)
        print('public_dataset', public_dataset)
    # if args.mode == 'fedda':
    #     if args.public_dataset > 0 and args.synthesize_mode == 'global':
    #         public_dataset = datasets[args.public_dataset-1]
    #         datasets.pop(args.public_dataset-1)
    #         print('public_dataset', public_dataset)
    #     elif args.public_dataset == 0 and args.synthesize_mode == 'local':
    #         pass
    #     else:
    #         print('There is a problem in synthesize mode')
    datasets = datasets[0:args.client_num]
    print('datasets', datasets)
    
    # prepare the data
    # trainsets, virtualsets, testsets, train_loaders, virtual_loaders, test_loaders = prepare_data(args,datasets,public_dataset)
    trainsets, virtualsets, testsets, generatsets = prepare_data(args,datasets,public_dataset, (256, 256))
    print(len(trainsets),len(virtualsets),len(testsets),len(generatsets))

                                   
    train_loaders = []
    test_loaders = []
    virtual_loaders = []
    generate_loaders = []
    adapt_test_loaders = []
    for sets in trainsets:
        train_loaders.append(torch.utils.data.DataLoader(sets, batch_size=args.batch, shuffle=True))
        print('size')
        print(len(sets))
    for sets in testsets:
        test_loaders.append(torch.utils.data.DataLoader(sets, batch_size=args.batch, shuffle=False))
    for sets in virtualsets:
        virtual_loaders.append(torch.utils.data.DataLoader(sets, batch_size=args.batch, shuffle=True))
    for sets in generatsets:
        generate_loaders.append(torch.utils.data.DataLoader(sets, batch_size=args.batch, shuffle=True))
    if args.synthesize_test:
        for sets in testsets:
            adapt_test_loaders.append(torch.utils.data.DataLoader(sets, batch_size=args.batch, shuffle=False))

    # for client_idx, train_loader in enumerate(train_loaders):
    #     iter_img = iter(train_loader)
    #     x, y = next(iter_img)
    #     for i in range(10):
    #         class_idx = np.argmax(y.numpy())
    #         plt.imshow(np.moveaxis(x[i].numpy(), 0, -1))
    #         plt.savefig('../images/' + str(datasets[client_idx]) + '_class_'+ str(y[i]) + '_' + str(i))\

    class_count = [0, 0]
    for idx in range(len(train_loaders)):
        num = [0, 0]
        print(datasets[idx])
        train_iter = iter(train_loaders[idx])
        for i in range(len(train_loaders[idx])):
            x, y = next(train_iter)
            num[1] += np.sum(y.numpy())
            num[0] += len(y.numpy()) - np.sum(y.numpy())
            # print(y.numpy())
            # print(np.argmax(y.numpy(), -1))
        print(num)
        class_count[0] += num[0]
        class_count[1] += num[1]

    
    # fig, axes = plt.subplots(4,len(datasets),figsize=(40,32))

    # federated setting
    client_num = args.client_num
    client_weights = [1 / client_num for i in range(client_num)]
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load('../snapshots/digits/{}'.format(args.mode.lower()))
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(models[test_idx], test_loader, loss_fun, device)
                print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
        else:
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(server_model, test_loader, loss_fun, device)
                print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
        exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        resume_iter = int(checkpoint['a_iter']) + 1
        # resume_iter = 21

        print('Resume training from epoch {}'.format(resume_iter))
    else:
        resume_iter = 0
    max_train_acc = 0
    max_test_acc = 0
    lr = args.lr
    # lr_schedulers =  [ReduceLROnPlateau(optimizer, factor=lr_factor, patience=lr_patience, threshold=lr_threshold) for optimizer]
    # start training
    patience = 0
    domain_discri = []
    domain_adv = []
    # print(models[0])
    features_dim = 512
    # for client_idx in range(client_num):
        # print('sanitycheck',trainsets[client_idx].path,
        #       len(trainsets[client_idx].images),
        #       len(virtualsets[client_idx].images),
        #       len(testsets[client_idx].images))

    # for client_idx in range(client_num):
    #     if args.uda_type == 'dann':
    #         domain_discri.append(DomainDiscriminator(in_feature=features_dim, hidden_size=1024).to(device))
    #         domain_adv.append((domain_discri[client_idx]).to(device))
    #     elif args.uda_type == 'cdan':
    #         domain_discri.append(DomainDiscriminator(features_dim * trainset_num_classes, hidden_size=1024).to(device))
    #         domain_adv.append(ConditionalDomainAdversarialLoss(domain_discri[client_idx], entropy_conditioning=False,
    #                                                            num_classes=trainset_num_classes,
    #                                                            features_dim=features_dim, randomized=False).to(device))
    loss_list = []
    for a_iter in range(resume_iter, args.iters):
        # optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        # if a_iter > 0:
        #     lr = args.lr / a_iter
        #     if patience == 4:
        #         server_model = Best_Global_model
        #         models = Best_local_models
        #         patience = 0
        #         args.param_cdan /= 5
        if args.mode.lower() == 'fedda' and a_iter > args.pre_iter:
            # optimizers = [optim.SGD(params=models[idx].parameters() + domain_discri[idx].parameters() , lr=lr) for idx in range(client_num)]
            optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
            for param in server_model.parameters():
                param.requires_grad = False
            server_model.eval()
        elif args.mode.lower() == 'fedvss':
            optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
            for param in server_model.parameters():
                param.requires_grad = False
            server_model.eval()
        else:
            optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]

        print("============ Train epoch {} ============".format(a_iter * args.wk_iters))
        # if args.log: logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters))
        # fig, axes = plt.subplots(4, 2, figsize=(24, 48))
        if args.mode.lower() == 'fedda' and a_iter > args.pre_iter and args.synthesize_mode != None:
            vir_datasets = []
            vir_labels = []
            ori_datasets = []
            ori_labels = []
            for client_idx,generate_loader in enumerate(generate_loaders):
                if args.synth_method == 'ce':
                    print('generating data for client', client_idx)
                    vir_dataset, vir_label, ori_dataset, ori_label = src_img_synth_ce(generate_loader, server_model,
                                                                                        args, device, 'train',
                                                                                        Synth_SAVE_PATH + '_train_' +
                                                                                        datasets[
                                                                                            client_idx] + '_' + str(
                                                                                            a_iter) + '.png', a_iter,
                                                                                        trainset_num_classes, wandb,
                                                                                        class_count)
                    vir_datasets.append(vir_dataset)
                    vir_labels.append(vir_label)
                    ori_datasets.append(ori_dataset)
                    ori_labels.append(ori_label)
                elif args.synth_method == 'admm':
                    print('generating data for client', client_idx)
                    vir_dataset, vir_label, ori_dataset, ori_label = src_img_synth_admm(generate_loader, server_model, args,device, 'train', Synth_SAVE_PATH+'_train_' +datasets[client_idx]+'_'+ str(a_iter) + '.png',a_iter, trainset_num_classes, wandb, class_count)
                    vir_datasets.append(vir_dataset)
                    vir_labels.append(vir_label)
                    ori_datasets.append(ori_dataset)
                    ori_labels.append(ori_label)

            # for client_idx in range(len(generate_loaders)):
            #     virtualsets[client_idx].images = ori_datasets[client_idx].detach().cpu().numpy()
            #     virtualsets[client_idx].labels = ori_labels[client_idx].detach().cpu().numpy()
            #     virtualsets[client_idx].synthesized = True
            #     virtual_loaders[client_idx] = torch.utils.data.DataLoader(virtualsets[client_idx], batch_size=args.batch, shuffle=True)

            # if (a_iter - 1) % args.save_every == 0:
            #     print('making first row plots')
            #     testset_vis = trainsets[0:client_num] 
            #     testloader_vis = train_loaders[0:client_num]
            #     for i in range(len(generate_loaders)):
            #         testset_vis.append(virtualsets[i])
            #         if (testset_vis[i].labels == testset_vis[i+client_num].labels).all():
            #             print('trueeee')
            #         testloader_vis.append(virtual_loaders[i])
            #     trans = fit_umap(models, testloader_vis, testset_vis, device, client_num+len(generate_loaders))
            #     print(len(testloader_vis))
            #     visualize_all(models, testloader_vis, testset_vis, axes[0, 0], axes[0, 1], device, client_num+len(generate_loaders), trans)
            #     plt.savefig(FIG_SAVE_PATH+ '_' + str(a_iter) + '.png')

            for client_idx in range(len(generate_loaders)):
                # vir_iter = iter(virtual_loaders[client_idx])
                # for i in range(len(vir_iter)):
                #     if i >2:
                #         break
                #     x,y = next(vir_iter)
                #     print(y)
                virtualsets[client_idx].images = vir_datasets[client_idx].detach().cpu()
                virtualsets[client_idx].labels = vir_labels[client_idx].detach().cpu()
                # print(virtualsets[client_idx].labels)
                virtualsets[client_idx].synthesized = True
                virtual_loaders[client_idx] = torch.utils.data.DataLoader(virtualsets[client_idx], batch_size=args.batch*10, shuffle=True)
                # vir_iter = iter(virtual_loaders[client_idx])
                # print('synthsynth')
                # for i in range(len(vir_iter)):
                #     if i >2:
                #         break
                #     x,y = next(vir_iter)
                #     print(y)

#             if (a_iter - 1) % args.save_every == 0:
#                 print('making second row plots')
#                 testset_vis = trainsets[0:client_num] 
#                 testloader_vis = train_loaders[0:client_num]
#                 for i in range(len(generate_loaders)):
#                     testset_vis.append(virtualsets[i])
#                     testloader_vis.append(virtual_loaders[i])
#                 trans = fit_umap(models, testloader_vis, testset_vis, device, client_num+len(generate_loaders))

#                 print(len(testloader_vis))
#                 visualize_all(models, testloader_vis, testset_vis, axes[1, 0], axes[1, 1], device, client_num+len(generate_loaders), trans)
#                 plt.savefig(FIG_SAVE_PATH+ '_' + str(a_iter) + '.png')


        for client_idx in range(client_num):
            model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
            if args.mode.lower() == 'fedprox':
                if a_iter > 0:
                    train_fedprox(args, wandb, server_model, model, train_loader, optimizer, loss_fun, client_num, device,client_idx,args.wk_iters)
                else:
                    train(args, wandb,model, train_loader, optimizer, loss_fun, client_num, device,client_idx,args.wk_iters)
            elif args.mode.lower() == 'fedda':
                if a_iter > args.pre_iter:
                    if args.synthesize_mode == 'local':
                        # train_uda(trg_loader=train_loader, src_loader=virtual_loaders[client_idx], trg_model=model,
                        #           domain_adv=domain_adv[client_idx], optimizer=optimizer, epoch=args.wk_iters, args=args,
                        #           device=device,wandb=wandb,client_idx=client_idx)
                        train_multi_datasets(args, wandb, model, [train_loader, virtual_loaders[client_idx]], optimizer,
                                             loss_fun, client_num, device, client_idx, args.wk_iters)
                    elif args.synthesize_mode == 'global':
                        # train(args, wandb, model, train_loader, optimizer, loss_fun, client_num, device,
                        #       client_idx, args.wk_iters)

                        train_multi_datasets(args, wandb,model, [train_loader, virtual_loaders[0]], optimizer, loss_fun, client_num, device,client_idx,args.wk_iters)

                        # train_uda(trg_loader=train_loader, src_loader=virtual_loaders[0], trg_model=model,
                        #           domain_adv=domain_adv[client_idx], optimizer=optimizer, epoch=args.wk_iters, args=args,
                        #           device=device,wandb=wandb,client_idx=client_idx)
                else:
                    train(args, wandb,model, train_loader, optimizer, loss_fun, client_num, device,client_idx,args.wk_iters)
            elif args.mode.lower() == 'fedvss':
                train_fedvss(args, wandb, server_model, model, train_loader, optimizer, loss_fun, client_num, device, client_idx,
                      args.wk_iters)
            else:
                train(args, wandb,model, train_loader, optimizer, loss_fun, client_num, device,client_idx,args.wk_iters)

            if args.synthesize_test:
                test_loss, test_acc = test(model, adapt_test_loaders[client_idx], loss_fun, device)
            else:
                test_loss, test_acc = test(model, test_loaders[client_idx], loss_fun, device)

            if args.log:
                metrics = {"Test_ACC_Local_" + str(client_idx): test_acc}
                wandb.log(metrics)

        # if (a_iter-1) % args.save_every == 0:
        #     print('making third row plots')
        #     testset_vis = trainsets[0:client_num] 
        #     testloader_vis = train_loaders[0:client_num]
        #     mult = 0
        #     if args.mode == 'fedda' and a_iter > args.pre_iter:
        #         for i in range(len(generate_loaders)):
        #             testset_vis.append(virtualsets[i])
        #             testloader_vis.append(virtual_loaders[i])
        #         mult = 1
        #     trans = fit_umap(models, testloader_vis, testset_vis, device, mult*len(generate_loaders) + client_num)
        #     visualize_all(models, testloader_vis, testset_vis, axes[2, 0], axes[2, 1], device, mult*len(generate_loaders) + client_num, trans)
        #     plt.savefig(FIG_SAVE_PATH+ '_' + str(a_iter) + '.png')
        # aggregation
        server_model, models = communication(args, server_model, models, client_weights, client_num, a_iter)
                
        
        
        if args.mode.lower() == 'fedda' and a_iter > args.pre_iter and args.synthesize_mode != None:
            for client_idx,generate_loader in enumerate(generate_loaders):
                if args.synth_method == 'ce':
                    pass
                elif args.synth_method == 'admm':
                    if args.synthesize_test:
                        vir_test_dataset, vir_test_label, ori_test_dataset, ori_test_label = src_img_synth_admm(test_loader, server_model, args,device, 'test', Synth_SAVE_PATH++'_test_' + datasets[client_idx]+'_'+ str(a_iter) + '.png',a_iter)
                        adapt_test_loaders[client_idx].images = vir_test_dataset.detach().cpu().numpy()
                        adapt_test_loaders[client_idx].labels = ori_test_label.detach().cpu().numpy()
                        adapt_test_loaders[client_idx].synthesized = True
        
        # if (a_iter-1) % args.save_every == 0:
        #     print('making forth row plots')
        #     testset_vis = trainsets[0:client_num] 
        #     testloader_vis = train_loaders[0:client_num]
        #     mult = 0
        #     if args.mode == 'fedda' and a_iter > args.pre_iter:
        #         for i in range(len(generate_loaders)):
        #             testset_vis.append(virtualsets[i])
        #             testloader_vis.append(virtual_loaders[i])
        #         mult = 1
        #     trans = fit_umap(models, testloader_vis, testset_vis, device, mult*len(generate_loaders) + client_num)
        #     visualize_all(models, testloader_vis, testset_vis, axes[3, 0], axes[3, 1], device, mult*len(generate_loaders) + client_num, trans)
        #     plt.savefig(FIG_SAVE_PATH+ '_' + str(a_iter) + '.png')
        

        # report after aggregation
        avg_train = 0
        avg_loss = 0
        for client_idx in range(client_num):
            model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
            if args.synthesize_test:
                train_loss, train_acc = test(model, virtual_loaders[client_idx], loss_fun, device)
            else:
                train_loss, train_acc = test(model, train_loader, loss_fun, device)
            avg_train += train_acc
            avg_loss += train_loss
            print(
                ' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx], train_loss, train_acc))
            if args.log:
                # logfile.write(
                    # ' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx], train_loss,
                    #                                                             train_acc))
                metrics = {"Train_ACC_" + str(client_idx): train_acc,
                           "Train_Loss_" + str(client_idx): train_loss}
                wandb.log(metrics)

        #seaborn.set()
        #loss_list.append(avg_loss)
        #plt.plot(range(len(loss_list)), loss_list)
        ## plt.legend('', loc='upper right')
        #plt.ylabel('Cross-entropy Loss')
        #plt.xlabel('Communication Round')
        #plt.savefig('loss_curve_' + str(args.wk_iters))
        #np.save('medical_loss_'+str(args.wk_iters), np.array(loss_list))


        if max_train_acc < avg_train / client_num:
            max_train_acc = avg_train / client_num
        if args.log:
            metrics = {"Train_AVG": avg_train / client_num}
            wandb.log(metrics)
            metrics = {"train_loss_avg": avg_loss / client_num}
            wandb.log(metrics)

        # start testing
        avg_test = 0
        for test_idx, test_loader in enumerate(test_loaders[0:client_num]):
            if args.synthesize_test:
                test_loss, test_acc = test(models[test_idx], adapt_test_loaders[test_idx], loss_fun, device)
            else:
                test_loss, test_acc = test(models[test_idx], test_loader, loss_fun, device)
           
            avg_test += test_acc
            print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasets[test_idx], test_loss, test_acc))
            if args.log:
                # logfile.write(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(datasets[test_idx], test_loss,
                #                                                                           test_acc))
                metrics = {"Test_ACC_" + str(test_idx): test_acc,
                           "Test_Loss_" + str(test_idx): test_loss}
                wandb.log(metrics)

        if max_test_acc < avg_test / client_num:
            max_test_acc = avg_test / client_num
            Best_Global_model = server_model
            Best_local_models = models
            patience = 0
        else:
            patience += 1

        if args.log:
            metrics = {"Test_AVG": avg_test / client_num}
            wandb.log(metrics)
        print('Maximum train accuracy average is:', max_train_acc)
        print('Maximum test accuracy average is:', max_test_acc)

        # Save checkpoint
        print(' Saving checkpoints to {}...'.format(SAVE_PATH))
        save = {}
        save['server_model'] =  server_model.state_dict()
        save['a_iter'] =  a_iter
        if args.mode.lower() == 'fedbn':
            for i in range(client_num):
                save['model_'+str(i)] = models[i].state_dict()
            torch.save(save, SAVE_PATH)
        else:
            torch.save(save, SAVE_PATH)
            
        # plt.close()

    # if log:
    #     logfile.flush()
    #     logfile.close()

