import os
import math
import tempfile
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms



from CustomData.my_dataset import MyDataSet
from CustomData.NewSplitdata import read_split_data
from TrainUtils.distributed_utils import init_distributed_mode, dist, cleanup
from TrainUtils.train_eval_utils import train_one_epoch, evaluate
import Utils
from MetricEval import ClassifierEvalMulticlass, ClassifierEvalBinary
from sklearn import metrics
import numpy as np

# ZhuanLi dpn+srm
from Model.ZhuanLi.ZhuanLi import SRMDPN_ZhuanLi

def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distributed_mode(args=args)

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    weights_path = args.weights
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")


    train_images_path, train_images_label = read_split_data(args.train_path, "train")
    val_images_path, val_images_label = read_split_data(args.val_path, "val")

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([
            # transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    # 实例化验证数据集
    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)
    # 实例化模型
    # 6-22 SRMDPN_ZhuanLi 中的相关实验 数据集8-1-1
    modelname = "SRMDPN_ZhuanLi-811"
    model = SRMDPN_ZhuanLi()

    model = model.to(device)

    # 如果存在预训练权重则载入
    if os.path.exists(weights_path):
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

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
        train_sampler.set_epoch(epoch)

        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        sum_num, y_true, y_score = evaluate(model=model,
                                            data_loader=val_loader,
                                            device=device)


        if rank == 0:

            auc_score = metrics.roc_auc_score(y_true, y_score[:, 1])
            print(auc_score)

            val_accuracy = sum_num / val_sampler.total_size
            print("[epoch {}] accuracy: {}".format(epoch, round(val_accuracy, 3)))

            # confusion matrix
            confusion_matrix = metrics.confusion_matrix(y_true, np.argmax(y_score, 1))
            # AP
            # ok_val_ap, ng_val_ap, mAP = utils.get_AP_metric(y_true, y_score)
            mulit_class_ap = ClassifierEvalMulticlass.compute_ap(y_true, y_score)
            ng_val_ap = mulit_class_ap[0]
            ok_val_ap = mulit_class_ap[1]
            mAP = (ng_val_ap + ok_val_ap) / 2

            ok_y_score = y_score[:, 1]

            ok_p_at_r = ClassifierEvalBinary.compute_p_at_r(y_true, ok_y_score, 1)

            ng_y_true = np.array(y_true).astype("bool")
            ng_y_true = (1 - ng_y_true).astype("int")
            ng_y_score = y_score[:, 0]
            ng_p_at_r = ClassifierEvalBinary.compute_p_at_r(ng_y_true, ng_y_score, 1)

            print(f'Acc: {val_accuracy:.3f}\t OK_AP：{ok_val_ap:.3f}\t NG_AP: {ng_val_ap:.3f}\t mAP: {mAP:.3f}')
            print(f'BEST Acc: {best_acc:.3f}\t OK_AP: {best_ok_ap:.3f}\t NG_AP: {best_ng_ap:.3f}\t mAP: {best_ap:.3f}')
            print("confusion_matrix  ne : au, ok : tp")
            print(confusion_matrix)
            print(mulit_class_ap)
            print(f'ok_p_at_r: {ok_p_at_r}, ng_p_at_r: {ng_p_at_r}')

            tags = ["accuracy", "ng_val_ap", "ok_val_ap", "learning_rate", "auc_score", "mean_loss"]
            tb_writer.add_scalar(tags[0], val_accuracy, epoch)
            tb_writer.add_scalar(tags[1], ng_val_ap, epoch)
            tb_writer.add_scalar(tags[2], ok_val_ap, epoch)
            tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)
            tb_writer.add_scalar(tags[4], auc_score, epoch)
            tb_writer.add_scalar(tags[5], mean_loss, epoch)

            save_name = f'{modelname}_{epoch}_acc_{val_accuracy:.4f}_p@r_{ng_p_at_r}.pth'
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

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.5)

    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)

    # 数据集所在根目录
    parser.add_argument('--train-path', type=str,
                        default=r'F:\experiment\CASIA811\train')
    parser.add_argument('--val-path', type=str,
                        default=r'F:\experiment\CASIA811\val')

    # resnet34 官方权重下载地址
    parser.add_argument('--weights', type=str, default="",
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    main(opt)
