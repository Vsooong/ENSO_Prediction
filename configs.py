from model.CNN.CNN2_4 import CNN2_4
from model.CNN.CNN2_3 import simpleSpatailTimeNN
from model.CNN.CNN2_5 import CNN2_5
from model.GraphCNN.graph_CNN import graphCNN
from model.ConvRNN.convlstm import convLSTM
from model.GraphRNN.AGCRN import AGCRN
from model.RNN.lstm import lstmNN
import torch
import os

args = {
    'model_list': {
        'CNN2_3': simpleSpatailTimeNN,
        'CNN2_4': CNN2_4,
        'CNN2_5': CNN2_5,
        'graphCNN': graphCNN,
        'convLSTM': convLSTM,
        'AGCRN': AGCRN,
        'lstmNN': lstmNN,
    },
    'pretrain': False,
    'n_epochs': 2000,
    'learning_rate': 1e-4,
    'batch_size': 12,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'grad_norm': False,
    'max_grad_norm': 5,
    'early_stop_patience': 20000,

    'cmip_data': r'D:\data\enso\meta_data\CMIP_train.nc',
    'cmip_label': r'D:\data\enso\meta_data\CMIP_label.nc',
    'sota_data': r'D:\data\enso\meta_data\SODA_train.nc',
    'sota_label': r'D:\data\enso\meta_data\SODA_label.nc',

    'cmip_sst': next(iter([i for i in [r'D:\data\enso\final_data\reshape_sst.npz',
                                       '/home/dl/GSW/data/enso/final_data/reshape_sst.npz']
                           if os.path.exists(i)]), None),
    'cmip_t300': next(iter([i for i in [r'D:\data\enso\final_data\reshape_t300.npz',
                                        '/home/dl/GSW/data/enso/final_data/reshape_t300.npz']
                            if os.path.exists(i)]), None),
    'cmip_ua': next(iter([i for i in [r'D:\data\enso\final_data\reshape_ua.npz',
                                      '/home/dl/GSW/data/enso/final_data/reshape_ua.npz']
                          if os.path.exists(i)]), None),
    'cmip_va': next(iter([i for i in [r'D:\data\enso\final_data\reshape_va.npz',
                                      '/home/dl/GSW/data/enso/final_data/reshape_va.npz']
                          if os.path.exists(i)]), None),
    'cmip_labels': next(iter([i for i in [r'D:\data\enso\final_data\reshape_nino.npz',
                                          '/home/dl/GSW/data/enso/final_data/reshape_nino.npz']
                              if os.path.exists(i)]), None),
    'soda_sst': next(iter([i for i in [r'D:\data\enso\final_data\soda_sst.npz',
                                       '/home/dl/GSW/data/enso/final_data/soda_sst.npz']
                           if os.path.exists(i)]), None),
    'soda_t300': next(iter([i for i in [r'D:\data\enso\final_data\soda_t300.npz',
                                        '/home/dl/GSW/data/enso/final_data/soda_t300.npz']
                            if os.path.exists(i)]), None),
    'soda_ua': next(iter([i for i in [r'D:\data\enso\final_data\soda_ua.npz',
                                      '/home/dl/GSW/data/enso/final_data/soda_ua.npz']
                          if os.path.exists(i)]), None),
    'soda_va': next(iter([i for i in [r'D:\data\enso\final_data\soda_va.npz',
                                      '/home/dl/GSW/data/enso/final_data/soda_va.npz']
                          if os.path.exists(i)]), None),
    'soda_label': next(iter([i for i in [r'D:\data\enso\final_data\soda_label.npz',
                                         '/home/dl/GSW/data/enso/final_data/soda_label.npz']
                             if os.path.exists(i)]), None),

    'path_list': [
        './tcdata/enso_final_test_data_B/',
        r'D:\data\enso\test样例_20210207_update\test样例',
        '/home/dl/GSW/data/enso/test样例_20210207_update/test样例',
    ]
}

if __name__ == '__main__':
    print(args['soda_sst'])
