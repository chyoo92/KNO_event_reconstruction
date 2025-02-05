#!/usr/bin/env python
import h5py
import sys, os
import numpy as np
import csv, yaml
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import ast
import subprocess
import math
from typing import Any, Optional
from torch import nn, Tensor
from torch.nn.functional import softplus

## torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## torch multi-gpu
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import datetime


sys.path.append("./module")

from model.allModel import *
from datasets import dataset_main


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', action='store', type=str) #### config file name
    parser.add_argument('-o', '--output', action='store', type=str) #### setting output folder name ==ex) vertex_test==
    parser.add_argument('--type', action='store', type=int, default=0) ##### 0 pid / 1 vertex / 2 energy / 3 direction
    parser.add_argument('--padding', action='store', type=int, default=0) #### 0 no padding / 1 inf padding
    parser.add_argument('--device', action='store', type=int, default=None) #### gpu device number 0,1,2,3.... only one gpu training
    parser.add_argument('--multi_device', type=int, default=2) #### when use multi_gpu training
    parser.add_argument('--transfer_learning', action='store', type=int, default=0) #### 0 first training / bigger than 1 transfer learning
    parser.add_argument('--rank_i', type=int, default=0) 

    #### model hyper parameter
    parser.add_argument('--fea', action='store', type=int, default=5) #### use 5 feature (charge, time, pmt_x, pmt_y, pmt_z)
    parser.add_argument('--cla', action='store', type=int, default=1) #### 1 : PID, energy / 3: vertex, direction
    parser.add_argument('--cross_head', action='store', type=int, default=1) #### Must be an absolute divisor of cross_dim  ///// If it's 1, it's mostly trained and you can default it to 1 and use it
    parser.add_argument('--cross_dim', action='store', type=int, default=64) #### Feature size in cross attention block.   /////  Recommended at least 128
    parser.add_argument('--self_head', action='store', type=int, default=8) #### Must be an absolute divisor of self_dim   /////  Recommended al least 8
    parser.add_argument('--self_dim', action='store', type=int, default=64) #### Feature size in cross attention block.   /////   Recommended at least 128
    parser.add_argument('--n_layers', action='store', type=int, default=3) #### The number of blocks in self-attention.   /////   5 or more is recommended
    parser.add_argument('--num_latents', action='store', type=int, default=200) #### The size of the latent array to be used for the cross-attention block. 
    parser.add_argument('--dropout_ratio', action='store', type=float, default=0.1) 



    #### training parameter
    parser.add_argument('--nDataLoaders', action='store', type=int, default=4)
    parser.add_argument('--epoch', action='store', type=int, default=300)
    parser.add_argument('--batch', action='store', type=int, default=25)
    parser.add_argument('--learningRate', action='store', type=float, default=0.001)
    parser.add_argument('--randomseed', action='store', type=int, default=12345)

    return parser



def main_one_gpu(args):
    
    dataset_module = dataset_main  #### main dataset code
    Dataset = dataset_module.NeuEvDataset

    device = 'cuda'

    #### config file load
    config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)
    if args.nDataLoaders: config['training']['nDataLoaders'] = args.nDataLoaders
    if args.epoch: config['training']['epoch'] = args.epoch
    if args.batch: config['training']['batch'] = args.batch
    if args.learningRate: config['training']['learningRate'] = args.learningRate
    if args.randomseed: config['training']['randomSeed'] = args.randomseed
    
    

    #### result folder
    if args.type == 0:
        result_path = 'result_pid/' + args.output
        if not os.path.exists(result_path): os.makedirs(result_path)
    elif args.type == 1:
        result_path = 'result_vertex/' + args.output
        if not os.path.exists(result_path): os.makedirs(result_path)
    elif args.type == 2:
        result_path = 'result_energy/' + args.output
        if not os.path.exists(result_path): os.makedirs(result_path)
    elif args.type == 3:    
        result_path = 'result_direction/' + args.output
        if not os.path.exists(result_path): os.makedirs(result_path)

    with open(result_path + '/' + args.output+'.txt', "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
    #### dataset 
    dset = Dataset()

    trnLoader, valLoader, _ = data_setting(args, config, dset)

    #### model load
    if args.transfer_learning == 0:
        model = perceiver_i(fea = args.fea, \
                        cla = args.cla, \
                        cross_head = args.cross_head, \
                        cross_dim = args.cross_dim, \
                        self_head = args.self_head, \
                        self_dim = args.self_dim, \
                        n_layers = args.n_layers, \
                        num_latents = args.num_latents, \
                        dropout_ratio = args.dropout_ratio, \
                        batch = config['training']['batch'], \
                        device = device)
    else:
        model = torch.load(result_path+'/model.pth',map_location='cuda')
        # model.load_state_dict(torch.load(result_path+'/model_state_dict_rt.pth',map_location='cuda'),strict=False)
        optm = torch.optim.Adam(model.parameters(), lr=config['training']['learningRate'])
        checkpoint_path = result_path + '/checkpoint_' + str(args.transfer_learning) + '.pth'  #### You can train by inserting a checkpoint file number at the desired point.
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        optm.load_state_dict(checkpoint['optimizer_state_dict'])


    
    torch.save(model, os.path.join(result_path, 'model.pth'))
    
    if args.type == 0:
        crit = torch.nn.BCEWithLogitsLoss().to(device)
    else:
        crit = LogCoshLoss().to(device)
    
        
    if args.transfer_learning == 0:    
        optm = torch.optim.Adam(model.parameters(), lr=config['training']['learningRate'])


    bestState, bestLoss = {}, 1e9
    if args.type == 0:
        train = {'loss':[], 'val_loss':[],'acc':[],'val_acc':[]}
    else:
        train = {'loss':[], 'val_loss':[]}
        
    nEpoch = config['training']['epoch']


    for epoch in range(nEpoch):

        trn_loss, trn_acc = 0., 0.
        nProcessed = 0

        ###############################################
        ################## training ###################
        ###############################################

        for i, batch_set in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, nEpoch))):
            model = model.cuda()

            if len(batch_set) == 9:
                pmt_q, pmt_t, vertex, particle, direction, energy, pmt_pos, fName, _ = batch_set
                
                padding_index = None
            else:
                pmt_q, pmt_t, vertex, particle, direction, energy, pmt_pos, fName, _, padding_index = batch_set

            pmts_q = pmt_q.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(device)
            pmts_t = pmt_t.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(device)
            pmt_pos = pmt_pos.to(device)

            if padding_index is not None:
                padding_index = padding_index.to(device)
            else:
                pass
            data = torch.cat([pmts_q,pmts_t],dim=2)

            if args.type == 0:
                label = particle.float().to(device=device).reshape(-1,1)
            elif args.type == 1:
                label = vertex.float().to(device=device).reshape(-1,3)
            elif args.type == 2:
                label = (energy.float().to(device=device).reshape(-1,1))/100 #### The reason for the division by 100 is that using the energy value as is does not train well.
            elif args.type == 3:
                label = direction.float().to(device=device).reshape(-1,3)


            if padding_index is not None:
                pred = model(data,pmt_pos,padding_index)

            else:
                pred = model(data,pmt_pos)
            
            loss = crit(pred, label)

            loss.backward()
            optm.step()
            optm.zero_grad()

            
            ibatch = len(vertex)
            nProcessed += ibatch
            trn_loss += loss.item()*ibatch
            
            if args.type == 0:
                trn_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.0, 1, 0))*ibatch
            
            del pmts_q, pmt_t, pmt_pos, data, vertex, particle, direction, energy, padding_index

        trn_loss /= nProcessed 
        print(trn_loss,'trn_loss')

        if args.type == 0:
            trn_acc  /= nProcessed
            print(trn_acc,'trn_acc')
            
        torch.save(model.state_dict(), os.path.join(result_path, 'model_state_dict_rt.pth'))


        ###############################################
        ################## validation #################
        ###############################################

        model.eval()
        val_loss, val_acc = 0., 0.
        nProcessed = 0
        with torch.no_grad():
            for i, batch_set in enumerate(tqdm(valLoader)):
                if len(batch_set) == 9:
                    pmt_q, pmt_t, vertex, particle, direction, energy, pmt_pos, fName, _ = batch_set
                    
                    padding_index = None
                else:
                    pmt_q, pmt_t, vertex, particle, direction, energy, pmt_pos, fName, _, padding_index = batch_set

                pmts_q = pmt_q.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(device)
                pmts_t = pmt_t.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(device)
                pmt_pos = pmt_pos.to(device)

                if padding_index is not None:
                    padding_index = padding_index.to(device)
                else:
                    pass
                data = torch.cat([pmts_q,pmts_t],dim=2)


                if args.type == 0:
                    label = particle.float().to(device=device).reshape(-1,1)
                elif args.type == 1:
                    label = vertex.float().to(device=device).reshape(-1,3)
                elif args.type == 2:
                    label = (energy.float().to(device=device).reshape(-1,1))/100 #### The reason for the division by 100 is that using the energy value as is does not train well.
                elif args.type == 3:
                    label = direction.float().to(device=device).reshape(-1,3)


                if padding_index is not None:
                    pred = model(data,pmt_pos,padding_index)

                else:
                    pred = model(data,pmt_pos)
                
                loss = crit(pred, label)
                
                ibatch = len(vertex)
                nProcessed += ibatch
                val_loss += loss.item()*ibatch
                if args.type == 0:
                    val_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.0, 1, 0))*ibatch


                del pmts_q, pmt_t, pmt_pos, data, vertex, particle, direction, energy, padding_index
                    
                    
            val_loss /= nProcessed
            print(val_loss,'val_loss')
            if args.type == 0:
                val_acc /= nProcessed
                print(val_acc,'val_acc')


            if bestLoss > val_loss:
                bestState = model.to('cpu').state_dict()
                bestLoss = val_loss
                torch.save(bestState, os.path.join(result_path, 'weight.pth'))


            # train['loss'].append(trn_loss)
            # train['val_loss'].append(val_loss)

            train['loss'] = [trn_loss]
            train['val_loss'] = [val_loss]

            if args.type == 0:
                # train['acc'].append(trn_acc)
                # train['val_acc'].append(val_acc)
                train['acc'] = [trn_acc]
                train['val_acc'] = [val_acc]

            file_path = os.path.join(result_path, 'train.csv')
            file_exists = os.path.isfile(file_path)

            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                # 파일이 새로 생성된 경우에만 헤더를 작성
                if not file_exists or os.path.getsize(file_path) == 0:
                    keys = train.keys()
                    writer.writerow(keys)

                # 데이터 추가
                keys = train.keys()
                for row in zip(*[train[key] for key in keys]):
                    writer.writerow(row)

            # with open(file_path, 'w') as f:
            #     writer = csv.writer(f)
            #     keys = train.keys()
            #     writer.writerow(keys)
            #     for row in zip(*[train[key] for key in keys]):
            #         writer.writerow(row)

        checkpoint = {
            'epoch': int(args.transfer_learning)+epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optm.state_dict(),
            'loss': loss.item(),
        }
        
        torch.save(checkpoint, f'{result_path}/checkpoint_{int(args.transfer_learning)+epoch+1}.pth')
        torch.save(model.to('cpu').state_dict(), f'{result_path}/weight_{int(args.transfer_learning)+epoch+1}.pth')
        
    bestState = model.to('cpu').state_dict()
    torch.save(bestState, os.path.join(result_path, 'weightFinal.pth'))





    return 0


#### The code below is the code we used when using multi_gpu.
#### It is not updated with the latest code and needs to be modified when used.

#     
# def main_multi_gpu(rank,args):
#     ### select dataset version
#     if args.datasetversion == 0:
#         dataset_module = dataset_main  #### main dataset code
#     elif args.datasetversion == 1:
#         dataset_module = dataset_test  #### test dataset code


#     Dataset = dataset_module.NeuEvDataset

#     #### multi gpu device distribute
#     local_gpu_id = init_for_distributed(rank,args)

#     #### config file load
#     config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)
#     if args.nDataLoaders: config['training']['nDataLoaders'] = args.nDataLoaders
#     if args.epoch: config['training']['epoch'] = args.epoch
#     if args.batch: config['training']['batch'] = args.batch
#     if args.learningRate: config['training']['learningRate'] = args.learningRate
#     if args.randomseed: config['training']['randomSeed'] = args.randomseed

    
#     #### result folder
#     if args.type == 0:
#         result_path = 'result_vtx/' + args.output
#         if not os.path.exists(result_path): os.makedirs(result_path)
#     elif args.type == 1:
#         result_path = 'result_pid/' + args.output
#         if not os.path.exists(result_path): os.makedirs(result_path)
#     elif args.type == 2:
#         result_path = 'result_dir/' + args.output
#         if not os.path.exists(result_path): os.makedirs(result_path)    
#     elif args.type == 3:
#         result_path = 'result_eng/' + args.output
#         if not os.path.exists(result_path): os.makedirs(result_path)    
#     with open(result_path + args.output, "w") as f:
#         for arg in vars(args):
#             f.write(f"{arg}: {getattr(args, arg)}\n")

#     #### dataset 
#     dset = Dataset()

#     trnLoader, valLoader, testLoader, train_sampler = data_setting(args, config, dset)
    

#     #### model load
#     if args.transfer_learning == 0:
#         model = perceiver_i(fea = args.fea, \
#                         cla = args.cla, \
#                         cross_head = args.cross_head, \
#                         cross_dim = args.cross_dim, \
#                         self_head = args.self_head, \
#                         self_dim = args.self_dim, \
#                         n_layers = args.n_layers, \
#                         num_latents = args.num_latents, \
#                         dropout_ratio = args.dropout_ratio, \
#                         batch = int(config['training']['batch']/args.world_size), \
#                         device = local_gpu_id)

#     elif args.transfer_learning == 1:
#         model = torch.load(result_path+'/model.pth',map_location='cuda')
#         model.load_state_dict(torch.load(result_path+'/model_module_state_dict_rt.pth',map_location='cuda'),strict=False)
#         optm = torch.optim.Adam(model.parameters(), lr=config['training']['learningRate'])
#         checkpoint_path = f'{result_path}/checkpoint.pth'
#         checkpoint = torch.load(checkpoint_path, map_location='cuda')
#         # model.load_state_dict(checkpoint['model_state_dict'],strict=False)
#         optm.load_state_dict(checkpoint['optimizer_state_dict'])
#     model = model.to(local_gpu_id)
#     # print(result_path)
#     if rank == 0:
#         torch.save(model, os.path.join(result_path, 'model.pth'))
    
        
#     model = DistributedDataParallel(module = model, device_ids=[local_gpu_id], find_unused_parameters=False)

#     #### Loss & optimizer 이거도 좀 세팅여러개 해야되는데
#     if args.loss_type == 0:
#         crit = LogCoshLoss().to(local_gpu_id)
#     elif args.loss_type == 1:
#         crit = torch.nn.BCEWithLogitsLoss().to(local_gpu_id)
#     elif args.loss_type == 2:
#         crit = torch.nn.L1Loss().to(local_gpu_id)
#     elif args.loss_type == 3:
#         crit = torch.nn.MSELoss().to(local_gpu_id)
#     elif args.loss_type == 4:
#         crit = BayesianRegressionLoss_vertex().to(local_gpu_id)
#     elif args.loss_type == 5:
#         crit = BayesianRegressionLoss_energy().to(local_gpu_id)
#     elif args.loss_type == 6:
#         crit = torch.nn.HuberLoss(reduction='mean', delta=1.0).to(local_gpu_id)

#     if args.transfer_learning == 0:
    
#         optm = torch.optim.Adam(model.parameters(), lr=config['training']['learningRate'])


#     bestState, bestLoss = {}, 1e9
#     # if args.type == 0:
#     #     train = {'loss':[], 'val_loss':[]}
#     # elif args.type == 1:
#     #     train = {'loss':[], 'val_loss':[],'acc':[],'val_acc':[]}
#     # elif args.type == 2:
#     #     train = {'loss':[], 'val_loss':[]}
#     # elif args.type == 3:
#     #     train = {'loss':[], 'val_loss':[]}

#     if args.type in [0, 2, 3]:
#         train = {'loss': [], 'val_loss': []}
#     elif args.type == 1:
#         train = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

#     nEpoch = config['training']['epoch']

#     for epoch in range(nEpoch):

#         trn_loss, trn_acc = 0., 0.
#         nProcessed = 0

#         ###############################################
#         ################## training ###################
#         ###############################################

#         train_sampler.set_epoch(epoch)
        
#         for i, batch_set in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, nEpoch))):
#             model = model.to(local_gpu_id)
#             model.train()
#             if len(batch_set) == 6:
#                 pmt_q, pmt_t, label, pmt_pos, _, _ = batch_set
#                 padding_index = None
#             else:
#                 pmt_q, pmt_t, label, pmt_pos, _, _, padding_index = batch_set

#             pmts_q = pmt_q.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(local_gpu_id)
#             pmts_t = pmt_t.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(local_gpu_id)
#             pmt_pos = pmt_pos.to(local_gpu_id)

#             if padding_index is not None:
#                 padding_index = padding_index.to(local_gpu_id)
#             else:
#                 pass           
#             data = torch.cat([pmts_q,pmts_t],dim=2)

#             label = label.float().to(device=local_gpu_id)

#             if args.vtx_1000 == 1:
#                 label = label/1000

#             # if args.type == 0: 
#             #     label = label.reshape(-1,3)
#             # elif args.type == 2: 
#             #     label = label.reshape(-1,3)
#             if args.type in [0, 2]:
#                 label = label.reshape(-1, 3)

#             if padding_index is not None:
#                 pred = model(data,pmt_pos,padding_index)
#             else:
#                 pred = model(data,pmt_pos)

#             if args.type in [1, 3]:
#                 pred = pred.reshape(-1)

            
#             loss = crit(pred, label)

#             loss.backward()
#             optm.step()
#             optm.zero_grad()


#             ibatch = len(label)
#             nProcessed += ibatch
#             trn_loss += loss.item()*ibatch
#             if args.type == 1:
#                 trn_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.0, 1, 0))*ibatch
#             del pmts_q, pmt_t, pmt_pos, data, label, pred

#         trn_loss /= nProcessed 
#         print(trn_loss,'trn_loss')
        
#         if args.type == 1:
#             trn_acc  /= nProcessed
#             print(trn_acc,'trn_acc')
#         if rank == 0:
#             torch.save(model.state_dict(), os.path.join(result_path, 'model_state_dict_rt.pth'))
#             torch.save(model.module.state_dict(), os.path.join(result_path, 'model_module_state_dict_rt.pth'))    

#         ###############################################
#         ################## validation #################
#         ###############################################

#         model.eval()
#         val_loss, val_acc = 0., 0.
#         nProcessed = 0
#         with torch.no_grad():
#             for i, batch_set in enumerate(tqdm(valLoader)):
#                 if len(batch_set) == 6:
#                     pmt_q, pmt_t, label, pmt_pos, _, _ = batch_set
#                     padding_index = None
#                 else:
#                     pmt_q, pmt_t, label, pmt_pos, _, _, padding_index = batch_set

#                 pmts_q = pmt_q.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(local_gpu_id)
#                 pmts_t = pmt_t.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(local_gpu_id)
#                 pmt_pos = pmt_pos.to(local_gpu_id)

#                 if padding_index is not None:
#                     padding_index = padding_index.to(local_gpu_id)
#                 else:
#                     pass
                
#                 data = torch.cat([pmts_q,pmts_t],dim=2)

#                 label = label.float().to(device=local_gpu_id)
#                 if args.vtx_1000 == 1:
#                     label = label/1000
#                 if args.type in [0, 2]: 
#                     label = label.reshape(-1,3)
#                 # elif args.type == 2: 
#                     # label = label.reshape(-1,3)
#                 if padding_index is not None:
#                     pred = model(data,pmt_pos,padding_index)
#                 else:
#                     pred = model(data,pmt_pos)
#                 if args.type in [1, 3]: 
#                     pred = pred.reshape(-1)
                
#                 loss = crit(pred, label)
                
#                 ibatch = len(label)
#                 nProcessed += ibatch
#                 val_loss += loss.item()*ibatch
#                 if args.type == 1:
#                     val_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.0, 1, 0))*ibatch


#                 del pmts_q, pmt_t, pmt_pos, data, label, pred
                    
                    
#             val_loss /= nProcessed
#             print(val_loss,'val_loss')
            
#             if args.type == 1:
#                 val_acc /= nProcessed
#                 print(val_acc,'val_acc')

#             if rank == 0:
#                 if bestLoss > val_loss:
#                     bestState = model.to('cpu').state_dict()
#                     bestLoss = val_loss
#                     torch.save(bestState, os.path.join(result_path, 'weight.pth'))

#                     model.to(local_gpu_id)
#                     torch.save(model.module.state_dict(), os.path.join(result_path, 'model_scrpited_min_val.pth'))   
            
            
#             train['loss'].append(trn_loss)
#             train['val_loss'].append(val_loss)

#             if args.type == 1:
#                 train['acc'].append(trn_acc)
#                 train['val_acc'].append(val_acc)

#             if rank == 0:
#                 with open(os.path.join(result_path, 'train.csv'), 'w') as f:
#                     writer = csv.writer(f)
#                     keys = train.keys()
#                     writer.writerow(keys)
#                     for row in zip(*[train[key] for key in keys]):
#                         writer.writerow(row)
#         checkpoint = {
#             'epoch': epoch+1,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optm.state_dict(),
#             'loss': loss.item(),
#         }
#         if rank==0:
#             torch.save(checkpoint, f'{result_path}/checkpoint_{epoch+1}.pth')
#             torch.save(model.to('cpu').state_dict(), f'{result_path}/weight_{epoch+1}.pth')


        
    

#     bestState = model.to('cpu').state_dict()
#     torch.save(bestState, os.path.join(result_path, 'weightFinal.pth'))
#     torch.save(model.module.state_dict(), os.path.join(result_path, 'model_final.pt h'))    




#     return 0



class LogCoshLoss(nn.Module):
    """Log-cosh loss function."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        weights: Optional[Tensor] = None,
        return_elements: bool = False,
    ) -> Tensor:
        diff = prediction - target
        elements = self._log_cosh(diff)
        if weights is not None:
            elements *= weights

        assert (
            elements.size(0) == target.size(0)
        ), "`_log_cosh` should return elementwise loss terms."

        return elements if return_elements else torch.mean(elements)

    @classmethod
    def _log_cosh(cls, x: Tensor) -> Tensor:
        return x + softplus(-2.0 * x) - np.log(2.0)
    

##########################################################
################## Setting data ##########################
##########################################################


def data_setting(args,config,dset):


    for sampleInfo in config['samples']:
        if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
        name = sampleInfo['name']
        dset.addSample(sampleInfo['path'], sampleInfo['label'], args.padding)
    dset.initialize() 


    #### split events
    lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
    lengths.append(len(dset)-sum(lengths))
    torch.manual_seed(config['training']['randomSeed'])
    trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)
    print(len(trnDset),len(valDset),len(testDset))

    
    if isinstance(args.device,int):

        kwargs = {'batch_size':config['training']['batch'],'num_workers':min(config['training']['nDataLoaders'],os.cpu_count()), 'pin_memory':True}

        trnLoader = torch.utils.data.DataLoader(trnDset, shuffle=True, **kwargs)
        valLoader = torch.utils.data.DataLoader(valDset, shuffle=False, **kwargs)
        testLoader = torch.utils.data.DataLoader(testDset, shuffle=False, **kwargs)
        torch.manual_seed(torch.initial_seed())

        return trnLoader, valLoader, testLoader    
    elif len(args.device) > 1:
        kwargs = {'batch_size':int(config['training']['batch']/args.world_size), 'num_workers':config['training']['nDataLoaders'], 'pin_memory':True}

        train_sampler = DistributedSampler(trnDset,shuffle=True)
        trnLoader = torch.utils.data.DataLoader(trnDset, shuffle=False, sampler=train_sampler, **kwargs)

        val_sampler = DistributedSampler(valDset,shuffle=False)
        valLoader = torch.utils.data.DataLoader(valDset, shuffle=False, sampler=val_sampler, **kwargs)

        testLoader = torch.utils.data.DataLoader(testDset, shuffle=False, **kwargs)
        torch.manual_seed(torch.initial_seed())

        return trnLoader, valLoader, testLoader, train_sampler



##########################################################
################## Setting for multi GPU #################
##########################################################

def init_for_distributed(rank,args):
    
    # 1. setting for distributed training
    args.rank = rank
    local_gpu_id = int(args.device[args.rank])
    torch.cuda.set_device(local_gpu_id)
    if args.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id),args.rank)

    # 2. init_process_group
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=args.world_size,
                            rank=args.rank,
                            timeout=datetime.timedelta(seconds=3600))

    # if put this function, the all processes block at all.
    torch.distributed.barrier()
    # convert print fn iif rank is zero
    setup_for_distributed(args.rank == 0)
    print(args,'--------------initialize--------------')
    return local_gpu_id


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


\

##########################################################
###################### main running ######################
##########################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser('training', parents=[get_args_parser()])
    args = parser.parse_args()


    if args.device is not None:
        args.device = args.device

        
        main_one_gpu(args)
    else:
        args.device = args.multi_device
        
        device_list = []
        for i in range(args.device):
            device_list.append(str(i))
        
        args.device = device_list
        args.world_size = len(args.device)
        

        
        mp.spawn(main_multi_gpu, args=(args,),nprocs=args.world_size,join=True)
    
    