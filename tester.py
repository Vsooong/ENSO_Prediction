import shutil
from configs import args
import torch
import os
import numpy as np
import zipfile

args['model_name'] = 'simpleSpatailTimeNN'


def norm(adj):
    # adj += np.eye(adj.shape[0]) # 为每个结点增加自环
    degree = np.array(adj.sum(1))  # 为每个结点计算度
    degree = np.diag(np.power(degree, -0.5))
    return degree.dot(adj).dot(degree)


def test(in_path='./tcdata/enso_round1_test_20210201/',
         out_path='result'):
    if not os.path.exists(in_path):
        for path in args['path_list']:
            if os.path.exists(path):
                in_path = path

    if os.path.exists(out_path):
        shutil.rmtree(out_path, ignore_errors=True)
    os.makedirs(out_path)

    test_sample_file = [os.path.join(in_path, i) for i in os.listdir(in_path) if i.endswith('.npy')]
    device = args['device']
    model = args['model_list'][args['model_name']]()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(current_dir, 'experiments', args['model_name'] + '.pth')
    model.load_state_dict(torch.load(save_dir, map_location=device))
    model.to(device)
    model.eval()

    for i in test_sample_file:
        data = np.load(i)
        data = np.nan_to_num(data)
        # adj = torch.tensor(norm(np.ones((4, 4))), dtype=torch.float).to(device)
        sst = torch.as_tensor(data[..., 0], dtype=torch.float).to(device).unsqueeze(0)
        t300 = torch.as_tensor(data[..., 1], dtype=torch.float).to(device).unsqueeze(0)
        ua = torch.as_tensor(data[..., 2], dtype=torch.float).to(device).unsqueeze(0)
        va = torch.as_tensor(data[..., 3], dtype=torch.float).to(device).unsqueeze(0)
        preds = model(sst, t300, ua, va)
        if len(preds) == 2:
            preds = preds[1]
        preds = preds.squeeze(0).cpu().detach().numpy()

        save_path = os.path.join(out_path, os.path.basename(i))
        np.save(file=save_path, arr=preds)
    make_zip(out_path, 'result.zip')


def make_zip(source_dir, output_filename):
    f = zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            f.write(os.path.join(dirpath, filename))
    f.close()


if __name__ == '__main__':
    test()
    # path = r'D:\data\enso\test样例_20210207_update\test样例\label/test_0144-01-12.npy'
    # from lib.metric import eval_score
    # data=np.expand_dims( np.load(path),axis=0)
    # print(data)
    # print(eval_score(data,data))
