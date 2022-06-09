# -*- coding: utf-8 -*-
from torch.utils.data.dataloader import DataLoader
import torch as t
from torchnet import meter
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tqdm import tqdm
import numpy as np

from config import DefaultConfig
from dataset import MCBenMal
from utlis.evaluations import evaluate
from models.SAHNet import SAHNet_ResNet18

opt = DefaultConfig()
def train(**kwargs):
    opt.parse(kwargs)
    model = SAHNet_ResNet18()
    if opt.mulitGPU:
        model = t.nn.DataParallel(model)
    if opt.use_gpu:
        model.cuda()
    train_data = MCBenMal(opt.train_data_root, train=True)
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True,
                                  # sampler=sampler,
                                  pin_memory=True,
                                  num_workers=opt.num_workers)
    val_data = MCBenMal(opt.validation_data_root, validation=True)
    val_dataloader = DataLoader(val_data, 1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=opt.num_workers)
    criterion_cls = t.nn.CrossEntropyLoss()
    criterion_att = t.nn.L1Loss(reduction='mean')
    lr = opt.lr

    if opt.mulitGPU:
        transformer_params = list(map(id, model.module.Dtransformer.parameters()))
        backbone_params = filter(lambda p: id(p) not in transformer_params,
                                 model.module.parameters())
        params_dict = [{'params': backbone_params}, {'params': model.module.Dtransformer.parameters(), 'lr': lr * 0.5}]
    else:
        transformer_params = list(map(id, model.Dtransformer.parameters()))
        backbone_params = filter(lambda p: id(p) not in transformer_params,
                                 model.parameters())
        params_dict = [{'params': backbone_params}, {'params': model.Dtransformer.parameters(), 'lr': lr * 0.5}]


    optimizer =t.optim.Adam(params_dict, lr=lr,weight_decay=opt.weight_decay)

    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)

    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()
        optimizer.zero_grad()

        for ii, (data, label) in tqdm(enumerate(train_dataloader)):

            input = Variable(data).float()
            target = Variable(label).long()
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            score, att_mask = model(input)
            loss = criterion_cls(score, target)+0.1*criterion_att(att_mask[0],att_mask[1])\
                   +0.1*criterion_att(att_mask[0],att_mask[2])+0.1*criterion_att(att_mask[1],att_mask[2])
            loss.backward()
            optimizer.step()
            model.zero_grad()

            loss_meter.add(loss.item())
            confusion_matrix.add(score.data, target.data)

        cm_value = confusion_matrix.value()
        train_acc = 100. * ((cm_value[0][0] + cm_value[1][1]) /
                            (cm_value.sum()))

        val_cm, Val_loss, roc_auc, accuracy, precision, recall, specificity, F1 = val(model, val_dataloader)
        print('Epoch: {}/{},Lr: {:.6f}\nTrain_Loss: {:.6f}, Val_Loss:{:.6f},TrainAccuracy: {:.2f}%,AUC: {:.4f}, ValAccuracy: {:.2f}%, '
              'precision: {:.2f}%, recall: {:.2f}%, specificity: {:.2f}%, F1: {:.2f}%'.format(
            epoch, opt.max_epoch,lr, loss_meter.value()[0], Val_loss,train_acc, roc_auc, accuracy, precision, recall, specificity, F1
        ))
        

def val(model, dataloader):
    model.eval()
    scores = np.array([0, 0])
    labels = np.array([0])
    criterion = t.nn.CrossEntropyLoss()
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = input.float()
        val_label = label.long()
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score, att_mask = model(val_input)
        loss = criterion(score, val_label)
        loss_meter.add(loss.item())
        confusion_matrix.add(score.data, val_label.long())
        score_tem = score.cpu().detach().numpy()
        label_tem = val_label.cpu().detach().numpy()
        label_tem = label_tem.reshape(label_tem.shape[0], 1)
        scores = np.vstack((scores, score_tem))
        labels = np.vstack((labels, label_tem))

    scores = np.delete(scores, 0, axis=0)
    labels = np.delete(labels, 0, axis=0)
    Val_loss = loss_meter.value()[0]
    model.train()

    # validation results
    fpr, tpr, roc_auc, accuracy, precision, recall, specificity, F1 = evaluate(confusion_matrix, scores, labels)

    return confusion_matrix, Val_loss, roc_auc, accuracy, precision, recall, specificity, F1


def test(**kwargs):
    opt.parse(kwargs)
    model = t.load(opt.load_model_path,map_location='cpu')
    model.eval()

    test_data = MCBenMal(opt.test_data_root, test=True)
    test_dataloader = DataLoader(test_data,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers)
    results = []
    scores = np.array([0, 0])
    labels = np.array([0])
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(test_dataloader):
        input,label,path = data
        with t.no_grad():
            input = input.float()
        if opt.use_gpu:
            input = input.cuda()
            label = label.cuda()
        score = model(input)
        confusion_matrix.add(score.data, label.long())
        score_tem = score.cpu().detach().numpy()
        label_tem = label.cpu().detach().numpy()
        label_tem = label_tem.reshape(label_tem.shape[0], 1)
        scores = np.vstack((scores, score_tem))
        labels = np.vstack((labels, label_tem))

        probability = t.nn.functional.softmax(score,dim = 1).data.tolist()
        batch_resluts = [(path,label, probability)
                         for path,label,probability in zip(path,label, probability)]
        results += batch_resluts
    write_csv(results, opt.result_file)
    scores = np.delete(scores, 0, axis=0)
    labels = np.delete(labels, 0, axis=0)
    fpr, tpr, roc_auc, accuracy, precision, recall, specificity, F1 = evaluate(confusion_matrix, scores, labels)
    print('AUC: {:.2f}, Accuracy: {:.2f}%, precision: {:.2f}%, recall: {:.2f}%,'
          'specificity: {:.2f}%, F1: {:.2f}%'.format(roc_auc, accuracy, precision, recall, specificity, F1
    ))
    return results


if __name__=='__main__':
    train()
    # test()
