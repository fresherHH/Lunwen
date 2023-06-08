import math
import os
import shutil

import numpy as np
import torch

import config
from MetricEval import ClassifierEvalBinary


def load_model(pth, cuda_index):
    """
    一机多卡load
    :return:
    """
    if cuda_index == -1:
        # load to cpu
        checkpoint = torch.load(config.test_pth, map_location=torch.device('cpu'))
    else:
        # load to cuda
        checkpoint = torch.load(pth, map_location=lambda storage, loc: storage.cuda())

    model = checkpoint.module
    return model


def adjust_lr(optimizer, cur_epoch, gamma=0.5, warm_up=5):
    cur_epoch += 1
    original_lr = config.lr
    if cur_epoch <= warm_up:
        new_lr = original_lr * cur_epoch / warm_up
    else:
        new_lr = original_lr * gamma * (1 + math.cos(cur_epoch / config.max_epoch * math.pi))

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def movefile(index, mode):
    in_file_path = f'/mnt/tmp/feng/kuozhankuang/fold_{index}/train/{mode}'
    out_file_path = f'/mnt/tmp/feng/kuozhankuang/fold_{index}/val/{mode}'
    if not os.path.exists(out_file_path):
        os.makedirs(out_file_path)
    for file in os.listdir(in_file_path):
        file_path = f'{in_file_path}/{file}'
        p = np.random.rand(1)[0]
        if p > 0.8:
            shutil.move(file_path, f'{out_file_path}/{file}')


def check_move_file():
    for index in [1]:
        for mode in ['ok', 'ng']:
            in_file_path = f'/mnt/tmp/feng/kuozhankuang/fold_{index}/train/{mode}'
            out_file_path = f'/mnt/tmp/feng/kuozhankuang/fold_{index}/val/{mode}'
            in_length = len(os.listdir(in_file_path))
            out_length = len(os.listdir(out_file_path))
            print(in_length / (in_length + out_length))


def get_data_count():
    from config import data_path
    for index in ['train', 'val', 'test']:
        for mode in ['Au', 'Tp']:
            count = len(os.listdir(f'{data_path}/{index}/{mode}'))
            print(index, mode, count)


def get_FOR_metric(y_true, y_score):
    threshold = y_score[y_true == 0].min(axis=0)[0]

    true_ok_ng_score = y_score[y_true == 1][:, 0]
    not_ok = true_ok_ng_score > threshold

    true_ok_length = len(y_true[y_true == 1])
    not_ok_length = len(not_ok[not_ok == True])

    final_metric_dict = {
        'threshold': threshold,
        'true=ok@pred=not_ok': not_ok_length,
        'true=ok@pred=ok': true_ok_length - not_ok_length,
        'all_ok': true_ok_length,
        'FOR': (true_ok_length - not_ok_length) / true_ok_length,
    }
    return final_metric_dict


def get_AP_metric(y_true, y_score):
    ok_y_score = y_score[:, 1]
    ok_val_ap = ClassifierEvalBinary.compute_ap(y_true, ok_y_score)

    ng_y_true = np.array(y_true).astype("bool")
    ng_y_true = (1 - ng_y_true).astype(np.int)
    ng_y_score = y_score[:, 0]
    ng_val_ap = ClassifierEvalBinary.compute_ap(ng_y_true, ng_y_score)

    mAP = (ok_val_ap + ng_val_ap) / 2

    return ok_val_ap, ng_val_ap, mAP


def write_log(log_file_name, ACC, AP, names):
    with open(f'./log/{log_file_name}', 'w', encoding='utf-8') as f:
        f.writelines(f"model: {config.model}\n")
        f.writelines(f"lr: {str(config.lr)}\n")
        f.writelines(f"momentum: {str(config.momentum)}\n")
        f.writelines(f"weight_decay: {str(config.weight_decay)}\n")
        f.writelines(f"batch_size: {str(config.batch_size)}\n")
        f.writelines(f"criterion: {config.criterion}\n")


        best_acc_epoch = list(map(str, ACC))
        best_ap_epoch = list(map(str, AP))

        f.writelines(f"best acc epoch: {' '.join(best_acc_epoch)}\n")
        f.writelines(f"best ap epoch: {' '.join(best_ap_epoch)}\n")
        f.writelines('\n'.join(names))


if __name__ == '__main__':
    get_data_count()
