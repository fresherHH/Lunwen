import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from CustomData.my_dataset import MyDataSet
from CustomData.NewSplitdata import read_split_data

from MetricEval import ClassifierEvalMulticlass, ClassifierEvalBinary
from sklearn import metrics
import numpy as np
from TrainUtils.train_eval_utils import evaluate

from Model.ResNet.ResNet34 import  resnext50_32x4d
from Model.LunWen.CMADPN68 import TranModel
from Model.ResNet.ResNet34 import ResNextTranModel
from Model.DPNet.T1 import Two_Stream_Netv1

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
               transforms.CenterCrop(256),
               transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )


    # load image
    # var_path = r'F:\experiment\CASIA2.0\train'
    var_path = r'F:\experiment\CASIA2\test'
    # var_path = r'C:\Users\HH\Desktop\Dataset\CASIA1\train'

    val_images_path, val_images_label = read_split_data(var_path, "test")

    # 实例化验证数据集
    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform)

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_data_set.collate_fn)


    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    # model = resnext50_32x4d(num_classes=2).to(device)
    # model = TranModel()

    # modelname = "ResNext50-811-sync"
    # model = ResNextTranModel()
    #
    #
    # 7-11 epoch=100, batch-size=8 time7-11 数据集是：8-2-0 sync
    modelname = "T1_Two_Stream_Netv1_100epo-sync"
    model = Two_Stream_Netv1(dropout=0.5)


    model = model.to(device)

    # load model weights
    weights_path = "F:\HHTemp\checkpoints\T1_Two_Stream_Netv1_100epoadam\T1_Two_Stream_Netv1_100epoadam_76_acc_0.9741.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)

    model = torch.load(weights_path, map_location=device)

    # prediction
    model.eval()
    with torch.no_grad():

        # validate
        sum_num, y_true, y_score = evaluate(model=model,
                                            data_loader=val_loader,
                                            device=device)

        auc_score = metrics.roc_auc_score(y_true, y_score[:, 1])
        print(auc_score)

        val_accuracy = sum_num / len(val_data_set)


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
        print("confusion_matrix  ne : au, ok : tp")
        print(confusion_matrix)
        print(mulit_class_ap)
        print(f'ok_p_at_r: {ok_p_at_r}, ng_p_at_r: {ng_p_at_r}')



if __name__ == '__main__':
    main()
