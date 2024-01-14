import os
import wfdb
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from wfdb  import processing
from resnet1d.LSTM import LSTM
from resnet1d.net1d import Net1D
from resnet1d.crnn1d import CRNN
from resnet1d.resnet1d import ResNet1D
from sklearn.model_selection import StratifiedKFold


def load_FRAG(path):
    data_list = []
    label_list = []
    dir_list = os.listdir(path)
    for f in tqdm(dir_list, desc='Processing'):
        if f.find('hea') == -1:
            continue
        f = os.path.splitext(f)[0]
        file = os.path.join(path, f)
        data = wfdb.rdrecord(file)
        data, d = wfdb.processing.resample_sig(data.p_signal.squeeze(1), 250, 125)
        data_mid = data[87:87 + 187]
        label = f.split('_')[2]
        label_list.append(label)
        data_list.append(data_mid)

    data_list = np.array(data_list)
    data_list = pd.DataFrame(data_list)
    label_list = pd.Series(label_list, name='label')

    return pd.concat([data_list, label_list], axis=1)


def load_MIT(path):
    mit = pd.read_csv(os.path.join(path, 'mitbih_train.csv'), header=None)
    return mit


def load_PTB(path):
    normal = pd.read_csv(os.path.join(path, 'ptbdb_normal.csv'), header=None)
    abnormal = pd.read_csv(os.path.join(path, 'ptbdb_abnormal.csv'), header=None)
    return normal, abnormal


def load_all_data(cv=False):
    mit = load_MIT('datasets/MITBIH')
    normal, abnormal = load_PTB('datasets/PTB')
    ph = load_FRAG('datasets/FRAG/all_frag')

    mit.columns = ph.columns
    appN = mit[mit['label'] == 0]
    appS = mit[mit['label'] == 1]
    appV = mit[mit['label'] == 2]
    normal.columns = ph.columns
    normal['label'] = 'N0'

    data_list = pd.concat([ph, normal, appS, appV, appN], ignore_index=True)

    if cv:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
        for i, (train_index, test_index) in enumerate(skf.split(data_list.iloc[:, 0:-1], data_list['label'])):
            data_list.iloc[test_index].to_pickle('datasets/data_split_{}.pkl'.format(i + 1))

    else:
        rand = pd.Series(np.random.randint(1, 11, size=len(data_list)), name='fold')
        split_set = pd.concat([data_list, rand], axis=1)
        val_fold = 9
        test_fold = 10
        # Train
        train = split_set.loc[split_set['fold'] < val_fold, : 'label']
        # Validate
        val = split_set.loc[split_set['fold'] == val_fold, : 'label']
        # Test
        test = split_set.loc[split_set['fold'] == test_fold, : 'label']
        return train, val, test


def load_Net1D(base_filters, filter_list, m_blocks_list, kernel_size, ordinal=True, path=None):
    model = Net1D(
        in_channels=1,
        base_filters=base_filters,
        ratio=1.0,
        filter_list=filter_list,
        m_blocks_list=m_blocks_list,
        kernel_size=kernel_size,
        stride=2,
        groups_width=1,
        verbose=False,
        n_classes=4 - int(ordinal),
        use_do=False)
    if path is not None:
        model.load_state_dict(torch.load(path))

    return model


def load_ResNet1D(path=None):
    model = ResNet1D(
        in_channels=1,
        base_filters=32,
        kernel_size=3,
        stride=2,
        verbose=False,
        n_block=3,
        n_classes=4,
        groups=1)
    if path is not None:
        model.load_state_dict(torch.load(path))

    return model


def load_CRNN(path=None):
    # in_channels, out_channels, n_len_seg, n_classes, device
    model = CRNN(1, 16, 187, 4, 'cuda')

    if path is not None:
        model.load_state_dict(torch.load(path))

    return model


def load_LSTM(path=None):
    model = LSTM()

    if path is not None:
        model.load_state_dict(torch.load(path))

    return model


if __name__ == '__main__':
    FD = load_FRAG('datasets/FRAG/all_frag')
    FD.to_pickle('triple_frag.pkl')
