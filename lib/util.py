import torch
import random
import numpy as np
import os


def init_seed(seed=1):
    '''
    Disable cudnn to maximize reproducibility
    '''
    # torch.cuda.cudnn_enabled = False
    # torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def print_model_parameters(model, only_num=False):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')


def get_memory_usage(device):
    allocated_memory = torch.cuda.memory_allocated(device) / (1024 * 1024.)
    cached_memory = torch.cuda.memory_cached(device) / (1024 * 1024.)
    return allocated_memory, cached_memory
    # print('Allocated Memory: {:.2f} MB, Cached Memory: {:.2f} MB'.format(allocated_memory, cached_memory))


# NINO3.4区（170°W-120°W，5°S-5°N）
def nino_index(ssta_1year):
    months = ssta_1year.shape(0)
    nino = []
    for i in range(months - 2):
        ssta1 = np.mean(ssta_1year[0, i, 10:13, 38:49].values)
        ssta2 = np.mean(ssta_1year[0, i + 1, 10:13, 38:49].values)
        ssta3 = np.mean(ssta_1year[0, i + 2, 10:13, 38:49].values)
        nino3_4 = (ssta1 + ssta2 + ssta3) / 3
        nino.append(nino3_4)
    return nino
