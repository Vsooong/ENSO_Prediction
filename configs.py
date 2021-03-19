from model.naive_CNN import simpleCNN, simpleSpatailTimeNN
import torch

args = {
    'model_name': 'simpleSpatailTimeNN',
    'model_list': {
        'simple_CNN': simpleCNN,
        'simpleSpatailTimeNN': simpleSpatailTimeNN
    },
    'pretrain': False,
    'n_epochs': 200,
    'learning_rate': 8e-5,
    'batch_size': 64,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'grad_norm': False,
    'max_grad_norm': 5,
    'early_stop_patience': 20,

    'cmip_data': r'D:\data\enso\meta_data\CMIP_train.nc',
    'cmip_label': r'D:\data\enso\meta_data\CMIP_label.nc',
    'sota_data': r'D:\data\enso\meta_data\SODA_train.nc',
    'sota_label': r'D:\data\enso\meta_data\SODA_label.nc',
    'path_list': [
        './tcdata/enso_round1_test_20210201/',
        r'D:\data\enso\test样例_20210207_update\test样例',
    ]
}
