import os
import time

import numpy as np
import torch
from sklearn import metrics
from torch import nn
from tqdm import tqdm


import Utils
from MetricEval import ClassifierEvalMulticlass, ClassifierEvalBinary
from torch.utils.tensorboard import SummaryWriter
from TrainUtils.distributed_utils import reduce_value, is_main_process


def trainer(device ,model, optimizer, criterion, scheduler, train_loader, val_loader, tqdm_length, epochs, log_flag=False):

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()

    best_acc = 0.5
    best_ap = 0
    best_FOR = 0
    best_ok_ap = 0
    best_ng_ap = 0

    best_ap_epoch = []
    best_acc_epoch = []
    best_FOR_epoch = []
    save_names = []

    for epoch in range(epochs):

        model.train()

        batch_avg_loss = 0
        optimizer.zero_grad()
        if is_main_process():
            bar = tqdm(enumerate(train_loader), total=tqdm_length)
        else:
            bar = tqdm(enumerate(train_loader), total=tqdm_length)

        for ii, (data, label) in bar:
            image = data.cuda()
            target = label.cuda()

            logits = model(image)
            loss = criterion(logits, target)
            loss.backward()

            cur_loss = loss.item()
            batch_avg_loss += cur_loss
            cur_lr = optimizer.state_dict()["param_groups"][0]["lr"]

            if is_main_process():
                bar.set_description(f'{epoch} loss:{cur_loss:.2e} lr:{cur_lr:.2e}')
            else:
                bar.set_description(f'{epoch} loss:{cur_loss:.2e} lr:{cur_lr:.2e}')

            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        # 等待所有进程计算完毕
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)


        val_accuracy, y_true, y_score = val(device, model, val_loader)



        # confusion matrix
        confusion_matrix = metrics.confusion_matrix(y_true, np.argmax(y_score, 1))
        # AP
        # ok_val_ap, ng_val_ap, mAP = utils.get_AP_metric(y_true, y_score)
        mulit_class_ap = ClassifierEvalMulticlass.compute_ap(y_true, y_score)
        ng_val_ap = mulit_class_ap[0]
        ok_val_ap = mulit_class_ap[1]
        mAP = (ng_val_ap + ok_val_ap) / 2
        # FOR
        final_metric_dict = Utils.get_FOR_metric(y_true, y_score)

        ok_y_score = y_score[:, 1]
        ok_p_at_r = ClassifierEvalBinary.compute_p_at_r(y_true, ok_y_score, 1)

        ng_y_true = np.array(y_true).astype("bool")
        ng_y_true = (1 - ng_y_true).astype(np.int)
        ng_y_score = y_score[:, 0]
        ng_p_at_r = ClassifierEvalBinary.compute_p_at_r(ng_y_true, ng_y_score, 1)

        print(f'Acc: {val_accuracy:.2f}\t OK_AP：{ok_val_ap:.2f}\t NG_AP: {ng_val_ap:.2f}\t mAP: {mAP:.2f}')
        print(f'BEST Acc: {best_acc:.2f}\t OK_AP: {best_ok_ap:.2f}\t NG_AP: {best_ng_ap:.2f}\t mAP: {best_ap:.2f}')
        print("confusion_matrix  ne : au, ok : tp")
        print(confusion_matrix)
        print(mulit_class_ap)
        print(f'ok_p_at_r: {ok_p_at_r}, ng_p_at_r: {ng_p_at_r}')
        print(final_metric_dict)

        tags = ["accuracy", "ng_val_ap", "ok_val_ap", "learning_rate"]
        tb_writer.add_scalar(tags[0], val_accuracy, epoch)
        tb_writer.add_scalar(tags[1], ng_val_ap, epoch)
        tb_writer.add_scalar(tags[2], ok_val_ap, epoch)
        tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        save_path = f'./checkpoints/'
        save_name = f'{epoch}_acc_{val_accuracy:.4f}_p@r_{ng_p_at_r}_FOR_{final_metric_dict["FOR"]:.4F}.pth'
        save_names.append(save_name)
        torch.save(model, f'{save_path}/{save_name}')

        if final_metric_dict['FOR'] > best_FOR:
            best_FOR = final_metric_dict['FOR']
            best_FOR_epoch.append(epoch)

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_acc_epoch.append(epoch)

        if mAP > best_ap:
            best_ap = mAP
            best_ap_epoch.append(epoch)

        best_ok_ap = max(ok_val_ap, best_ok_ap)
        best_ng_ap = max(ng_val_ap, best_ng_ap)

     

    if log_flag:
        cur_time = time.strftime('%m%d_%H_%M')
        log_file_name = f"Model_{cur_time}.txt"
        Utils.write_log(log_file_name, best_FOR_epoch, best_acc_epoch, best_ap_epoch, save_names)


@torch.no_grad()
def val(device, model, dataloader):
    model.eval()

    correct = 0
    total = 0

    y_true = np.array([])
    y_score = np.zeros(shape=(1, 2))

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(dataloader)
    else:
        data_loader = tqdm(dataloader)

    for step, (x, y) in enumerate(data_loader):
        x = x.cuda()
        y = y.cuda()

        output = model(x)
        _, predicted = torch.max(output.data, 1)

        softmax = nn.functional.softmax
        s_pred = softmax(output, dim=1)


        y_true = np.append(y_true, y.data.cpu().numpy())
        y_score = np.concatenate([y_score, s_pred.data.cpu().numpy()], axis=0)



        total += y.size(0)
        correct += (predicted == y).sum().item()

        # 等待所有进程计算完毕
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

    assert y_true.shape[0] == y_score.shape[0] - 1

    return correct / total, y_true, y_score[1:]
