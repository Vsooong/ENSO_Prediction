from lib.util import init_seed, print_model_parameters
from data_loader import load_train_data, load_val_data
from torch.utils.data import DataLoader, Subset
import torch
from copy import deepcopy
from lib.metric import eval_score
import os
import torch.nn as nn
from configs import args
import numpy as np

args['model_name'] = 'convLSTM'
current_dir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(current_dir, '../../experiments', args['model_name'] + '.pth')


def train():
    init_seed(11)
    indices = torch.randperm(1700)[:150]
    train_numerical = np.array([1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 15]) - 1
    val_numerical = np.array([3, 6, 9, 12]) - 1
    train_datasets = [Subset(load_train_data('cmip', which_num=num), indices) for num in train_numerical]
    valid_datasets = [Subset(load_val_data('cmip', which_num=num), indices) for num in val_numerical]
    train_loaders = [DataLoader(train_dataset, batch_size=args['batch_size']) for train_dataset in train_datasets]
    valid_loaders = [DataLoader(valid_dataset, batch_size=args['batch_size']) for valid_dataset in valid_datasets]

    device = args['device']
    model = args['model_list'][args['model_name']]()
    if args['pretrain'] and os.path.exists(save_dir):
        model.load_state_dict(torch.load(save_dir, map_location=device))
        print('load model from:', save_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
    loss_fn = nn.MSELoss()

    model.to(device)
    loss_fn.to(device)
    print_model_parameters(model)

    best_score = float('-inf')
    not_improved_count = 0

    for i in range(args['n_epochs']):
        model.train()
        loss_epoch = 0
        for cmip_num, train_loader in enumerate(train_loaders):
            loss_numodel = 0
            for step, (sst, t300, ua, va, label, sst_label) in enumerate(train_loader):
                sst = sst.to(device).float()
                t300 = t300.to(device).float()
                ua = ua.to(device).float()
                va = va.to(device).float()
                # sst_label = sst_label.to(device).float()
                optimizer.zero_grad()
                label = label.to(device).float()
                # output, preds = model(sst, t300, ua, va)
                preds = model(sst, t300, ua, va)
                loss1 = loss_fn(preds, label)
                # loss2 = loss_fn(output, sst_label)
                # loss = loss1 + loss2
                loss1.backward()
                loss_numodel += loss1.item()
                if args['grad_norm']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

                optimizer.step()
                del preds, loss1
            loss_epoch += loss_numodel
            print('numerical model {} loss: {}'.format(cmip_num, loss_numodel))

        model.eval()
        y_true, y_pred = [], []
        for valid_loader in valid_loaders:
            for step, (sst, t300, ua, va, label, sst_label) in enumerate(valid_loader):
                sst = sst.to(device).float()
                t300 = t300.to(device).float()
                ua = ua.to(device).float()
                va = va.to(device).float()
                label = label.to(device).float()
                preds = model(sst, t300, ua, va)
                y_pred.append(preds.detach())
                y_true.append(label.detach())
                del preds

        y_true = torch.cat(y_true, axis=0)
        y_pred = torch.cat(y_pred, axis=0)
        sco = eval_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        print('Epoch: {}, Train Loss: {}, Valid Score: {}'.format(i + 1, loss_epoch, sco))
        if sco > best_score:
            best_score = sco
            not_improved_count = 0
            best_state = True
        else:
            not_improved_count += 1
            best_state = False

        if not_improved_count == args['early_stop_patience']:
            print("Validation performance didn\'t improve for {} epochs. "  "Training stops.".format(
                args['early_stop_patience']))
            break
        if best_state:
            best_model = deepcopy(model.state_dict())
            torch.save(best_model, save_dir)
            # torch.save(model, '../user_data/ref.pkl')
            print('Model saved successfully:', save_dir)


if __name__ == '__main__':
    train()
