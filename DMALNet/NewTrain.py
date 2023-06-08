import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

import Utils
from MetricEval import ClassifierEvalMulticlass, ClassifierEvalBinary
from sklearn import metrics

from CustomData.my_dataset import MyDataSet

from CustomData.NewSplitdata import read_split_data
from TrainUtils.train_eval_utils import train_one_epoch, evaluate


#model
from Model.Xception.model_core import Two_Stream_Net
from Model.DPNet.CADPN1 import TranModelatten, TranModelattenv1
from Model.DPNet.SRMDPN import SRMDPNv1, SRMDPNv2, SRMDPNv3
from Model.DPNet.DPN import TranModel
from Model.DPNet.V1 import TransferModel, SRMDPNv4, SRMDPNv2

# 3-30
# from Model.DPNet.T1 import Two_Stream_Net, Two_Stream_Netv1

# resnext50相关实验 6-21
from Model.ResNet.ResNet34 import ResNextTranModel

# ZhuanLi dpn+srm
from Model.ZhuanLi.ZhuanLi import SRMDPN_ZhuanLi

from Model.ZhuanLi.DPN68Self import TransferModel


# 小论文相关实验 数据集：casia2.0 比例8-1-1, epoch=200 时间：
# from Model.Xception.model_core import Two_Stream_Net
# 小论文相关实验 数据集: casia2.0 比例8-1-1，epoch=200 时间7-4
from Model.DPNet.T1 import Two_Stream_Netv1

#小论文相关实验：
# from Model.LunWen.CMADPN68 import TranModel
# from Model.LunWen.DualDPN68 import Two_Stream_Net, Two_Stream_Net_1


#小论文将所有的BN换成LN
# from Model.LunWen.Tmp import TranModel

from Model.LunWen.RESXNET50 import RESXNET50


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()


    train_images_path, train_images_label = read_split_data(args.train_path, "train")
    val_images_path, val_images_label = read_split_data(args.val_path, "val")



    data_transform = {
        "train": transforms.Compose([
                                     # transforms.RandomResizedCrop(256),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
                                   # transforms.Resize(256),
                                   # transforms.CenterCrop(256),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )}

    # 实例化训练数据集
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    # 实例化验证数据集
    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_data_set.collate_fn)

    # 如果存在预训练权重则载入

    # modelname = "Two_Stream_Net"
    # model = Two_Stream_Net()                    #Xception论文中的网络模型  batches = 4
    # model = TranModelatten()                  #DPN四个模块的Cross attention


    # modelname = "DPN"
    # model = TranModel()


    # 3-30 在这个模型里面主要是添加了srm 模块的信息， 最后将两个fea进行特征融合看一下效果怎么样
    # modelname = "T1_Two_Stream_Net"
    # model = Two_Stream_Net(dropout=args.dropout)

    # # 3-31 在前面的基础上加了srm中间件
    # modelname = "T1_Two_Stream_Netv1"
    # model = Two_Stream_Netv1(dropout=args.dropout)
    #



    # 6-21 ResNext50_32-4d DPN68中的resnext 相关实验
    # modelname = "ResNext50_32-4d-811"
    # model = ResNextTranModel()


    # # Two-streamNet-811 : epoch=200, batch-size=4 time7-3 这是论文里面的CVPR
    # modelname = "Two-streamNet-811"
    # model = Two_Stream_Net()

    # # "DPN_Two_Stream_Netv1-811": epoch=200, batch-size=8 time7-4
    # modelname = "DPN_Two_Stream_Netv1-811"
    # model = Two_Stream_Netv1(dropout=args.dropout)
    #
    #
    # "CMADPN68-811": epoch=200, batch-size=8 time7-4
    #  runs DPN68+LN+811 best-acc 94.6%
    # modelname = "CMADPN68-811"
    # model = TranModel()

    # # "DualDPN688-811": epoch=200, batch-size=8 time7-9
    # #  runs DualDPN688-811  log Model_DualDPN688-811.txt
    # modelname = "DualDPN688-811"
    # model = Two_Stream_Net()

    # "DualDPN_Two_Stream_Net_1": epoch=200, batch-size=8 time7-10 为什么batch=16显存不够
    #  runs:   DualDPN688-DATA82      log:
    # modelname = "DualDPN_Two_Stream_Net_1"
    # model = Two_Stream_Net_1()


    # 7-11 epoch=100, batch-size=8 time7-11 数据集是：8-2-0 截止目前最好的结果
    # modelname = "T1_Two_Stream_Netv1_100epo"
    # model = Two_Stream_Netv1(dropout=args.dropout)

    # 7-11 epoch=100, batch-size=8 time7-11 数据集是：8-2-0 sync
    # modelname = "T1_Two_Stream_Netv1_100epo-nist16"
    # model = Two_Stream_Netv1(dropout=args.dropout)
    #

    # 7-11 epoch=100, batch-size=8 time7-11 数据集是：8-2-0 截止目前最好的结果
    modelname = "T1_Two_Stream_Netv1_100DVMM"
    model = Two_Stream_Netv1(dropout=args.dropout)


    # modelname = "Two_stream_NNET"
    # model = Two_Stream_Net()

    model = model.to(device)
    # auc 0.9872716382150344
    # weights_path = r"F:\HHTemp\checkpoints\T1_Two_Stream_Netv1_100casia1\T1_Two_Stream_Netv1_100casia1_86_acc_0.9483.pth"
    # assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    #
    # model = torch.load(weights_path, map_location=device)






    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # optimizer = optim.Adam(pg, lr=args.lr, weight_decay=0)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.5
    best_ap = 0
    best_FOR = 0
    best_ok_ap = 0
    best_ng_ap = 0

    best_ap_epoch = []
    best_acc_epoch = []
    best_FOR_epoch = []
    save_names = []
    save_path = f'./checkpoints/' + modelname
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(args.epochs):

        # train


        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        sum_num, y_true, y_score = evaluate(model=model,
                           data_loader=val_loader,
                           device=device)


        auc_score = metrics.roc_auc_score(y_true, y_score[:, 1])
        print(auc_score)

        val_accuracy = sum_num / len(val_data_set)
        print("[epoch {}] accuracy: {}".format(epoch, round(val_accuracy, 3)))

        # confusion matrix
        confusion_matrix = metrics.confusion_matrix(y_true, np.argmax(y_score, 1))
        # AP
        # ok_val_ap, ng_val_ap, mAP = utils.get_AP_metric(y_true, y_score)
        mulit_class_ap = ClassifierEvalMulticlass.compute_ap(y_true, y_score)
        ng_val_ap = mulit_class_ap[0]
        ok_val_ap = mulit_class_ap[1]
        mAP = (ng_val_ap + ok_val_ap) / 2

        #
        # ok_y_score = y_score[:, 1]
        #
        # ok_p_at_r = ClassifierEvalBinary.compute_p_at_r(y_true, ok_y_score, 1)
        #
        # ng_y_true = np.array(y_true).astype("bool")
        # ng_y_true = (1 - ng_y_true).astype("int")
        # ng_y_score = y_score[:, 0]
        # ng_p_at_r = ClassifierEvalBinary.compute_p_at_r(ng_y_true, ng_y_score, 1)




        print(f'Acc: {val_accuracy:.3f}\t OK_AP：{ok_val_ap:.3f}\t NG_AP: {ng_val_ap:.3f}\t mAP: {mAP:.3f}')
        print(f'BEST Acc: {best_acc:.3f}\t OK_AP: {best_ok_ap:.3f}\t NG_AP: {best_ng_ap:.3f}\t mAP: {best_ap:.3f}')
        print("confusion_matrix  ne : au, ok : tp")
        print(confusion_matrix)
        print(mulit_class_ap)
        # print(f'ok_p_at_r: {ok_p_at_r}, ng_p_at_r: {ng_p_at_r}')


        tags = ["accuracy", "ng_val_ap", "ok_val_ap", "learning_rate", "auc_score", "mean_loss"]
        tb_writer.add_scalar(tags[0], val_accuracy, epoch)
        tb_writer.add_scalar(tags[1], ng_val_ap, epoch)
        tb_writer.add_scalar(tags[2], ok_val_ap, epoch)
        tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[4], auc_score, epoch)
        tb_writer.add_scalar(tags[5], mean_loss, epoch)

        # save_name = f'{modelname}_{epoch}_acc_{val_accuracy:.4f}_p@r_{ng_p_at_r}.pth'
        save_name = f'{modelname}_{epoch}_acc_{val_accuracy:.4f}.pth'
        save_names.append(save_name)

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_acc_epoch.append(epoch)
            torch.save(model, f'{save_path}/{save_name}')

        if mAP > best_ap:
            best_ap = mAP
            best_ap_epoch.append(epoch)

        best_ok_ap = max(ok_val_ap, best_ok_ap)
        best_ng_ap = max(ng_val_ap, best_ng_ap)

        log_file_name = f"Model_{modelname}.txt"
        Utils.write_log(log_file_name, best_acc_epoch, best_ap_epoch, save_names)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)               #100 #20
    parser.add_argument('--batch-size', type=int, default=8)             # 16
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--dropout',type=float, default=0.5)


    # parser.add_argument('--train-path', type=str,
    #                     default=r'C:\Users\HH\Desktop\Dataset\CASIA1\train')
    # parser.add_argument('--val-path', type=str,
    #                     default=r'C:\Users\HH\Desktop\Dataset\CASIA1\val')

    parser.add_argument('--train-path', type=str,
                        default=r'C:\Users\HH\Desktop\Dataset\ImSpliceDataset1\train')
    parser.add_argument('--val-path', type=str,
                        default=r'C:\Users\HH\Desktop\Dataset\ImSpliceDataset1\val')

    # parser.add_argument('--train-path', type=str,
    #                     default=r'C:\Users\HH\Desktop\Dataset\Sync\train')
    # parser.add_argument('--val-path', type=str,
    #                     default=r'C:\Users\HH\Desktop\Dataset\Sync\val')

    # parser.add_argument('--train-path', type=str,
    #                     default=r'F:\experiment\CASIA2.0\train')
    # parser.add_argument('--val-path', type=str,
    #                     default=r'F:\experiment\CASIA2.0\val')


    # parser.add_argument('--train-path', type=str,
    #                     default=r'F:\experiment\CASIA811\train')
    # parser.add_argument('--val-path', type=str,
    #                     default=r'F:\experiment\CASIA811\val')

    parser.add_argument('--weights', type=str, default=r"checkpoints/T1_Two_Stream_Netv1_100epo/T1_Two_Stream_Netv1_100epo_94_acc_0.9682.pth",
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
