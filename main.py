import os
import random
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, ConfusionMatrixDisplay, \
    roc_curve, auc, accuracy_score

import loader
from CAM import cam_test


class TestDataset(Dataset):

    def __init__(self, data):
        self.label = data.iloc[:, -1]
        self.data = data.iloc[:, :-1]
        self.label = self.label_trans(self.label)

        self.data = np.array(self.data, dtype='float64')
        self.label = np.array(self.label)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index])

    def __len__(self):
        return len(self.data)

    def label_trans(self, label):
        label_ret = []
        # 原始的ECG结果及风险级别的字典
        ecg_risk_levels = {
            "N": 0,
            "Ne": 1,
            0: 2,
            "N0": 3,  # 插入的新行
            "AFIB": 4,
            "SVTA": 5,
            "SBR": 6,
            "BI": 7,
            "NOD": 8,
            "BBB": 9,
            1: 10,
            "VTLR": 11,
            "B": 12,
            "HGEA": 13,
            "VER": 14,
            2: 15,
            "VTHR": 16,
            "VTTdP": 17,
            "VFL": 18,
            "VF": 19,
            "MI": 20,
            3: 21
        }

        for l in label:
            label_ret.append(ecg_risk_levels[l])

        distribute = pd.Series(label_ret)
        print(distribute.value_counts())

        return label_ret


class MyDataset(Dataset):

    def __init__(self, data, ord=True):
        self.label = data.iloc[:, -1]
        self.data = data.iloc[:, :-1]
        self.label = self.label_trans(self.label)
        self.label = self.alter_label(self.label, ord)

        self.data = np.array(self.data, dtype='float64')
        self.label = np.array(self.label)

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float),
                torch.tensor(self.label[index], dtype=torch.float))

    def __len__(self):
        return len(self.data)

    def label_trans(self, label):
        label_ret = []
        # 原始的ECG结果及风险级别的字典
        ecg_risk_levels = {
            0: ["N", "Ne", 'Normal', 'N0', '0', 0],
            1: ["AFIB", "SVTA", "SBR", "BI", "NOD", '1', 1, "BBB"],  # 1: Supraventricular ectopic beats
            2: ["VTLR", "B", "HGEA", "VER", '2', 2],  # 2: Ventricular ectopic beats
            3: ["VTHR", "VTTdP", "VFL", "VF", 'MI', '3', 3]
        }

        # 构建通过ECG英文缩写获取风险级别的字典
        ecg_abbr_to_risk = {}
        for risk, ecg_list in ecg_risk_levels.items():
            for ecg_abbr in ecg_list:
                ecg_abbr_to_risk[ecg_abbr] = risk

        # 打印通过ECG英文缩写获取风险级别的字典
        print(ecg_abbr_to_risk)

        for l in label:
            label_ret.append(ecg_abbr_to_risk[l])

        distribute = pd.Series(label_ret)
        print(distribute.value_counts())

        return label_ret


    def alter_label(self, label, ord=True):
        # ordinal
        if ord:
            y_return = []
            for y in label:
                y_bin = []
                for i in range(3):
                    y_bin.append(int(y > i))
                y_return.append(y_bin)
            return y_return
        else:
            # classfication
            y_return = []
            for y in label:
                y_bin = [0, 0, 0, 0]
                y_bin[y] = 1
                y_return.append(y_bin)
            return y_return


def my_eval(gt, pred):
    res = []
    res.append(mean_absolute_error(gt, pred))
    return np.array(res)


def cal_pred_res(prob):
    test_pred = []
    for i, item in enumerate(prob):
        tmp_label = []
        tmp_label.append(1 - item[0])
        tmp_label.append(item[0] - item[1])
        tmp_label.append(item[1] - item[2])
        tmp_label.append(item[2])
        test_pred.append(tmp_label)
    return test_pred


def cross_validation(model, k, structure,writer, still=True, train=None, val=None, ord=True):

    if(still):
        datas = []
        for i in range(1, 11):
            if i != k:
                datas.append(pd.read_pickle(data_path+'data_split_{}.pkl'.format(i)))
        train = pd.concat(datas, ignore_index=True)
        val = pd.read_pickle(data_path+'data_split_{}.pkl'.format(k))

    dataset = MyDataset(train, ord)
    dataset_val = MyDataset(val, ord)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func1 = nn.BCELoss()
    step = 0

    for epoch in tqdm(range(n_epoch), desc="epoch"):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, drop_last=False, shuffle=True)

        # train
        model.train()
        prog_iter = tqdm(dataloader, desc="Training")
        for batch_idx, batch in enumerate(prog_iter):
            input_x, input_y = tuple(t.to(device) for t in batch)
            input_x = input_x.reshape(batch[0].shape[0], 1, -1)
            pred = model(input_x)

            loss = loss_func1(pred, input_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('{}-fold/train'.format(k), loss.item(), step)
            step += 1
        scheduler.step(epoch)

        # save model
        if epoch == (n_epoch - 1):
            model_save_path = os.path.join(model_path,
                                           '{}-{}-fold_epoch_{}_params_file.pkl'.format(structure, k, epoch))
            torch.save(model.state_dict(), model_save_path)

        # val
        model.eval()
        prog_iter_val = tqdm(dataloader_val, desc="Validation")
        val_pred_prob = []
        val_labels = []
        with (torch.no_grad()):
            for batch_idx, batch in enumerate(prog_iter_val):
                input_x, input_y = tuple(t.to(device) for t in batch)
                input_x = input_x.reshape(batch[0].shape[0], 1, -1)

                pred = model(input_x)

                pred = pred.cpu().data.numpy()
                if ord:
                    pred = cal_pred_res(pred)
                val_pred_prob.append(pred)
                val_labels.append(input_y.cpu().data.numpy())

        val_labels = np.concatenate(val_labels)
        val_pred_prob = np.concatenate(val_pred_prob)

        all_pred = np.argmax(val_pred_prob, axis=1)
        if ord:
            all_gt = np.sum(val_labels, axis=1)
        else:
            all_gt = np.argmax(val_labels, axis=1)

        writer.add_scalar('{}-fold/Acc'.format(k), accuracy_score(all_gt, all_pred), epoch)
        writer.add_scalar('{}-fold/val'.format(k), my_eval(all_pred, all_gt).item(), epoch)

    return all_gt, all_pred


def test_res(model, ord=True, path=None):
    test_set = MyDataset(test, ord=ord)
    dataloader_test = DataLoader(test_set, batch_size=1, drop_last=False, shuffle=False)
    prog_iter_test = tqdm(dataloader_test, desc="Testing")

    model.eval()
    if path != None:
        model.load_state_dict(torch.load(path))
    diff_x = []
    diff_y = []

    labels = [[], [], [], []]
    y = [[], [], [], []]

    dense_in = []

    def forward_hook(module, input, output):
        dense_in.append(input[0])

    target_layer = model.get_submodule('dense')
    forward_handle = target_layer.register_forward_hook(forward_hook)

    with torch.no_grad():
        for batch_idx, batch in enumerate(prog_iter_test):
            input_x, input_y = tuple(t.to("cpu") for t in batch)
            input_x = input_x.reshape(batch[0].shape[0], 1, -1)
            pred = model(input_x)
            pred_prob = pred.cpu().data.numpy()
            if ord:
                pred_prob = cal_pred_res(pred_prob)

            predc = np.argmax(pred_prob, axis=1).item()
            input_y = input_y.cpu().data.numpy()
            if ord:
                label = np.sum(input_y, axis=1).item()
            else:
                label = np.argmax(input_y, axis=1).item()
            diff_y.append([predc, label])

            for i in range(4):
                y[i].append(pred_prob[0][i])
                labels[i].append(label == i)

    p = pd.DataFrame(diff_y)
    p.columns = ['pred', 'gt']
    if not on_sever:
        con_mat = confusion_matrix(p['gt'], p['pred'])
        con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=0)[np.newaxis, :]  # 归一化
        con_mat_norm = np.around(con_mat_norm, decimals=2)
        annot = pd.DataFrame(con_mat_norm).applymap(lambda x: f"{x}")
        annot += pd.DataFrame(con_mat).applymap(lambda x: f"\n({x})")
        sns.heatmap(con_mat_norm, annot=annot, fmt="s", cmap='Blues').set(xlabel="Predict", ylabel="Truth")
        plt.show()

    print(classification_report(p['gt'], p['pred'], digits=6))

    for i in range(4):
        fpr, tpr, thresholds = roc_curve(labels[i], y[i], pos_label=1)
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
    return diff_x, diff_y, dense_in


def case_study(model, label, target, path=None):
    test_case = test[test['label'] == label]
    input_x = test_case.iloc[:, :-1]
    input_x = np.array(input_x, dtype='float64')
    input_x = torch.tensor(input_x, dtype=torch.float)
    model.eval()
    if path != None:
        model.load_state_dict(torch.load(path))

    with torch.no_grad():
        input_x = input_x.reshape(-1, 1, 187)
        pred = model(input_x)
        pred_prob = pred.cpu().data.numpy()
        pred_prob = cal_pred_res(pred_prob)
        pred = np.argmax(pred_prob, axis=1)
        res = pd.value_counts(pred)
        print(res)

    mask = (pred == target)
    show_res = pred[mask]
    show_case = input_x[mask]
    for i in range(min(len(show_res), 50)):
        cam_test(model, show_case[i], (show_res[i], target), 'stage_list.4.block_list.0.conv1.conv',
                 (label + i.__str__()), show_overlay=True)


if __name__ == '__main__':
    # Configs:
    test_flag = False
    new_distri = False
    on_sever = False
    ord = False
    comment = "CRNN_O"
    n_epoch = 20
    batch_size = 16
    base_filters = 32
    kernel_size = 8
    filter_list = [64, 128, 256, 512, 1024]
    m_blocks_list = [1, 1, 1, 1, 1]

    random.seed(0)
    torch.manual_seed(0)
    model_path = '/data/0shared/limingfei/models/' if on_sever else 'models/'
    data_path = '/data/0shared/limingfei/' if on_sever else 'datasets/processed/10f-all/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, 5))
    time = datetime.datetime.now().strftime("%F-%T").replace(":", "-")
    save_path = f'./logs/{time}-{comment}/'
    # loader.load_Net1D(base_filters, filter_list, m_blocks_list,kernel_size, False)
    if test_flag:

        test = pd.read_pickle('datasets/processed/10f-all/data_split_1.pkl')
        signals,results,features = test_res(loader.load_CRNN(),False, 'models/CRNN-1-fold_epoch_19_params_file.pkl')

        torch.save(signals, 'CRNN_sig.pth')
        torch.save(results, 'CRNN_res.pth')
        torch.save(features, 'CRNN_fea.pth')

    else:

        for k in range(1,11):

            model = loader.load_CRNN()
            writer = SummaryWriter(save_path)
            gt, pred = cross_validation(model,
                                        k,"CRNN_O",writer, ord=ord)
            cr = classification_report(gt, pred, digits=5)
            print(cr)
            with open(save_path+"class_report.txt", "a", encoding="utf-8") as f:
                print(f.write(cr))
            test = pd.read_pickle(data_path+f'data_split_{k}.pkl')
            test_res(model.to('cpu'),ord)

