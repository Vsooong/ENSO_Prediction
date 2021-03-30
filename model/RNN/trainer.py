from lib.util import init_seed, print_model_parameters, norm
from data_loader import load_train_data, load_val_data
from torch.utils.data import DataLoader, Subset
import torch
from copy import deepcopy
from lib.metric import eval_score
import os
import torch.nn as nn
from configs import args
import numpy as np

args['model_name'] = 'lstmNN'
args['batch_size'] = 64
current_dir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(current_dir, '../../experiments', args['model_name'] + '.pth')


def train():
    init_seed(111)
    split_num = 720
    train_num = 1600
    val_num = 100
    train_indices = torch.randperm(1700)[:train_num]
    val_indices = torch.randperm(1700)[:val_num]
    train_numerical = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14])
    val_numerical = np.array([6])
    # train_numerical = np.array([])
    # val_numerical = np.array([])

    train_datasets = [Subset(load_train_data('cmip', which_num=num), train_indices) for num in train_numerical]
    train_datasets.append(load_train_data('soda', split_num=split_num))
    print('Training Samples: {}'.format(len(train_numerical) * train_num + split_num))
    valid_datasets = [Subset(load_val_data('cmip', which_num=num), val_indices) for num in val_numerical]
    valid_datasets.append(load_val_data('soda', split_num=split_num + 60))
    print('Validation Samples: {}'.format(len(val_numerical) * val_num + 1200 - split_num))

    train_loaders = [DataLoader(train_dataset, batch_size=args['batch_size']) for train_dataset in train_datasets]
    valid_loaders = [DataLoader(valid_dataset, batch_size=args['batch_size']) for valid_dataset in valid_datasets]

    device = args['device']
    model = args['model_list'][args['model_name']]()
    print_model_parameters(model)

    if args['pretrain'] and os.path.exists(save_dir):
        model.load_state_dict(torch.load(save_dir, map_location=device))
        print('load model from:', save_dir)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'])
    if args['lr_decay']:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in args['lr_decay_step']]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_decay_steps,
                                                            gamma=args['lr_decay_rate'])
    else:
        lr_scheduler = None
    loss_fn = nn.MSELoss().to(device)
    # loss_fn = score_loss

    model.to(device)

    best_score = float('-inf')
    not_improved_count = 0

    for i in range(args['n_epochs']):
        model.train()
        loss_epoch = 0
        for train_loader in train_loaders:
            for step, (sst, t300, ua, va, label, month, sst_label) in enumerate(train_loader):
                sst = sst.to(device).float()
                t300 = t300.to(device).float()
                ua = ua.to(device).float()
                va = va.to(device).float()
                # month = month[:, :1].to(device).long()
                month = month.to(device).long()
                label = label.to(device).float()
                optimizer.zero_grad()

                preds = model(sst, t300, ua, va, month)
                loss = loss_fn(preds, label)
                loss.backward()
                loss_epoch += loss.item()
                if args['grad_norm']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

                optimizer.step()
                del preds, loss
        if args['lr_decay']:
            lr_scheduler.step()
        model.eval()
        y_true, y_pred = [], []
        for valid_loader in valid_loaders:
            for step, (sst, t300, ua, va, label, month, sst_label) in enumerate(valid_loader):
                sst = sst.to(device).float()
                t300 = t300.to(device).float()
                ua = ua.to(device).float()
                va = va.to(device).float()
                month = month.to(device).long()
                label = label.to(device).float()
                preds = model(sst, t300, ua, va, month)
                y_pred.append(preds.detach())
                y_true.append(label.detach())
                del preds

        y_true = torch.cat(y_true, axis=0)
        y_pred = torch.cat(y_pred, axis=0)
        sco = eval_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
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
