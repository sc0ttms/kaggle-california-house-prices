# -*- coding: utf-8 -*-

import os
import zipfile
import pandas as pd
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader
from config import Config


def pre_dataset(path):
    # extract dataset
    extract_dir = os.path.splitext(path)[0]
    fp = zipfile.ZipFile(path, 'r')
    fp.extractall(extract_dir)

    # read dataset
    train_dataset = pd.read_csv(extract_dir + '/train.csv')
    test_dataset = pd.read_csv(extract_dir + '/test.csv')

    test_id = test_dataset.iloc[:, 0]

    # normlize dataset
    all_data = pd.concat(
        (train_dataset.iloc[:, 1:], test_dataset.iloc[:, 1:]))
    all_data = all_data.select_dtypes(exclude='object')
    all_data = all_data.apply(lambda x: ((x - x.mean()) / x.std()))
    all_data = all_data.fillna(0)
    all_data = all_data.drop(['Sold Price'], axis=1)

    # split dataset
    n_train = train_dataset.shape[0]
    train_set = all_data[:n_train].values.astype(np.float32)
    test_set = all_data[n_train:].values.astype(np.float32)
    train_labels = np.reshape(
        train_dataset['Sold Price'].values.astype(np.float32), (-1, 1))
    return train_set, train_labels, test_set, test_id


def get_k_fold_data(n_k_fold, k, train_set, train_labels):
    assert n_k_fold > 1
    fold_size = train_set.shape[0] // n_k_fold
    X_train, y_train, X_valid, y_valid = None, None, None, None
    for i in range(n_k_fold):
        idx = slice(i*fold_size, (i+1)*fold_size)
        X_part, y_part = train_set[idx, :], train_labels[idx]
        if i == k:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = np.concatenate([X_train, X_part], 0)
            y_train = np.concatenate([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


class CHP_Dataset(Dataset):
    def __init__(self, X, y=None, mode='train'):
        super().__init__()

        self.X = X
        self.y = y
        self.mode = mode

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.mode in ['train', 'valid']:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


if __name__ == '__main__':
    device = paddle.get_device()
    paddle.set_device(device)
    print(f'device {device}')

    config = Config()
    config.device = device

    dataset_dir = 'C:/lbt/ML/datasets/kaggle_california_house_prices.zip'
    train_set, train_labels, test_set, _ = pre_dataset(dataset_dir)
    X_train, y_train, X_valid, y_valid = get_k_fold_data(
        config.n_k_fold, 1, train_set, train_labels)

    train_dataset = CHP_Dataset(X_train, y_train, mode='train')
    train_iter = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=config.drop_last)

    for i, (X, Y) in enumerate(train_iter()):
        print(i, X.shape, Y.shape)

    test_dataset = CHP_Dataset(test_set, mode='test')
    test_iter = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=config.drop_last)

    for i, (X) in enumerate(test_iter()):
        print(i, X.shape)
    pass
