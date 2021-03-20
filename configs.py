from model.CNN.naive_CNN import simpleSpatailTimeNN
from model.GraphCNN.graph_CNN import graphCNN
from model.ConvRNN.convlstm import convLSTM
import torch
import os

args = {
    'model_list': {
        'simpleSpatailTimeNN': simpleSpatailTimeNN,
        'graphCNN': graphCNN,
        'convLSTM': convLSTM,
    },
    'pretrain': False,
    'n_epochs': 2000,
    'learning_rate': 2e-3,
    'batch_size': 8,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'grad_norm': False,
    'max_grad_norm': 5,
    'early_stop_patience': 20000,

    'cmip_data': r'D:\data\enso\meta_data\CMIP_train.nc',
    'cmip_label': r'D:\data\enso\meta_data\CMIP_label.nc',
    'sota_data': r'D:\data\enso\meta_data\SODA_train.nc',
    'sota_label': r'D:\data\enso\meta_data\SODA_label.nc',

    'cmip_sst': [i for i in [r'D:\data\enso\final_data\cmip_sst.npz',
                             '/home/dl/Public/GSW/data/enso/final_data/cmip_sst.npz']
                 if os.path.exists(i)][0],
    'cmip_t300': [i for i in [r'D:\data\enso\final_data\cmip_t300.npz',
                              '/home/dl/Public/GSW/data/enso/final_data/cmip_t300.npz']
                  if os.path.exists(i)][0],
    'cmip_ua': [i for i in [r'D:\data\enso\final_data\cmip_ua.npz',
                            '/home/dl/Public/GSW/data/enso/final_data/cmip_ua.npz']
                if os.path.exists(i)][0],
    'cmip_va': [i for i in [r'D:\data\enso\final_data\cmip_va.npz',
                            '/home/dl/Public/GSW/data/enso/final_data/cmip_va.npz']
                if os.path.exists(i)][0],
    'cmip_labels': [i for i in [r'D:\data\enso\final_data\cmip_label.npz',
                                '/home/dl/Public/GSW/data/enso/final_data/cmip_label.npz']
                    if os.path.exists(i)][0],
    'soda_sst': [i for i in [r'D:\data\enso\final_data\soda_sst.npz',
                             '/home/dl/Public/GSW/data/enso/final_data/soda_sst.npz']
                 if os.path.exists(i)][0],
    'soda_t300': [i for i in [r'D:\data\enso\final_data\soda_sst.npz',
                              '/home/dl/Public/GSW/data/enso/final_data/soda_t300.npz']
                  if os.path.exists(i)][0],
    'soda_ua': [i for i in [r'D:\data\enso\final_data\soda_sst.npz',
                            '/home/dl/Public/GSW/data/enso/final_data/soda_ua.npz']
                if os.path.exists(i)][0],
    'soda_va': [i for i in [r'D:\data\enso\final_data\soda_sst.npz',
                            '/home/dl/Public/GSW/data/enso/final_data/soda_va.npz']
                if os.path.exists(i)][0],
    'soda_label': [i for i in [r'D:\data\enso\final_data\soda_sst.npz',
                               '/home/dl/Public/GSW/data/enso/final_data/soda_label.npz']
                   if os.path.exists(i)][0],

    'path_list': [
        './tcdata/enso_round1_test_20210201/',
        r'D:\data\enso\test样例_20210207_update\test样例',
    ]
}

if __name__ == '__main__':
    print(args['soda_ua'])
