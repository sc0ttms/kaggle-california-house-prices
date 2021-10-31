# -*- coding: utf-8 -*-

import os
import pandas as pd
import paddle
from paddle.io import DataLoader
from config import Config
from dataset import CHP_Dataset, pre_dataset
from model import MLP


def save_pref(test_id, preds, file):
    df = pd.DataFrame({'Id': test_id, 'Sold Price': preds})
    df.to_csv(file, index=False)


def test(model, test_iter, config):
    model.eval()
    y_hat = []
    for i, X in enumerate(test_iter()):
        with paddle.no_grad():
            y_hat.append(round(model(X).detach().cpu().item(), 3))
    return y_hat


if __name__ == '__main__':
    device = paddle.get_device()
    paddle.set_device(device)
    print(f'device {device}')

    config = Config()
    config.device = device

    dataset_dir = '/home/aistudio/data/data114434/kaggle_california_house_prices.zip'
    _, _, test_set, test_id = pre_dataset(dataset_dir)

    test_dataset = CHP_Dataset(test_set, mode='test')
    test_iter = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers)

    model = MLP(test_set.shape[1], config)
    model_dir = os.path.join(os.getcwd(), 'models',
                             'model_2_0.27014.pdparams')
    ckpt = paddle.load(model_dir)
    model.set_state_dict(ckpt)

    y_hat = test(model, test_iter, config)

    # save y_hat
    test_dir = os.path.join(os.getcwd(), 'test')
    os.makedirs(test_dir, exist_ok=True)
    save_pref(test_id, y_hat, os.path.join(
        test_dir, 'california_house_prices_preds.csv'))
