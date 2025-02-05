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

sys.path.append("./module")

from model.allModel import *
from datasets import dataset_main


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', action='store', type=str)
    parser.add_argument('-o', '--output', action='store', type=str)
    parser.add_argument('--type', action='store', type=int, default=0) ####### 0 vertex / 1 pid
    parser.add_argument('--padding', action='store', type=int, default=0)
    parser.add_argument('--device', action='store', type=int, default=None)
    parser.add_argument('--multi_device', type=int, default=2)
    parser.add_argument('--rank_i', type=int, default=0)
    parser.add_argument('--cla', action='store', type=int, default=3)

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

    # device = 'cuda'
    device = torch.device('cuda:'+str(args.device))
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
    elif args.type == 1:
        result_path = 'result_vertex/' + args.output
    elif args.type == 2:
        result_path = 'result_energy/' + args.output
    elif args.type == 3:    
        result_path = 'result_direction/' + args.output
    #### dataset 
    dset = Dataset()

    _, _, testLoader = data_setting(args, config, dset)
    
    #### model load
    
    model = torch.load(result_path+'/model.pth',map_location=device)
    model.load_state_dict(torch.load(result_path+'/weight.pth',map_location=device),strict=False)
    
    model = model.cuda(device)
    
    ###############################################
    ################## test #######################
    ###############################################


    labels, preds, fnames, file_events = [], [], [], []
    model.eval()

    for i, batch_set in enumerate(tqdm(testLoader)):
        if len(batch_set) == 9:
            pmt_q, pmt_t, vertex, particle, direction, energy, pmt_pos, fName, file_event = batch_set
            
            padding_index = None
        else:
            pmt_q, pmt_t, vertex, particle, direction, energy, pmt_pos, fName, file_event, padding_index = batch_set

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

        if args.type == 0: 
            pred = pred.reshape(-1)
            pred = torch.sigmoid(pred)

        labels.extend([x.item() for x in label.view(-1)])
        preds.extend([x.item() for x in pred.view(-1)])
        
        fnames.extend([x.item() for x in np.array(fName)])
        file_events.extend([x.item() for x in np.array(file_event)])

        del pmts_q, pmt_t, pmt_pos, data, vertex, particle, direction, energy, padding_index


    
    df = pd.DataFrame({'prediction':preds, 'label':labels})
    df2 = pd.DataFrame({'fname':fnames,'file_events':file_events})
    
    fPred = result_path+'/' + args.output + '.csv'
    df.to_csv(fPred, index=False)

    fName = result_path+'/name_' + args.output + '.csv'
    df2.to_csv(fName)


    del preds, labels, fnames



    return 0






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

    
    kwargs = {'batch_size':config['training']['batch'],'num_workers':min(config['training']['nDataLoaders'],os.cpu_count()), 'pin_memory':True}

    trnLoader = torch.utils.data.DataLoader(trnDset, shuffle=True, **kwargs)
    valLoader = torch.utils.data.DataLoader(valDset, shuffle=False, **kwargs)
    testLoader = torch.utils.data.DataLoader(testDset, shuffle=False, **kwargs)
    torch.manual_seed(torch.initial_seed())

    return trnLoader, valLoader, testLoader    


##########################################################
###################### main running ######################
##########################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser('training', parents=[get_args_parser()])
    args = parser.parse_args()


    
    args.device = args.device

    # with open(args.output, "w") as f:
    #     for arg in vars(args):
    #         f.write(f"{arg}: {getattr(args, arg)}\n")
    
    main_one_gpu(args)

        
