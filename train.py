# -*- coding: utf-8 -*-

import os
import paddle
from paddle.io import DataLoader
from timer import Timer
from config import Config
from dataset import CHP_Dataset, pre_dataset, get_k_fold_data
from model import MLP


def train(model, train_iter, valid_iter, config):
    n_epochs = config.n_epochs
    n_early_stop = config.n_early_stop

    model_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(model_dir, exist_ok=True)

    optimizer = getattr(
        paddle.optimizer, config.optimizer)(
            parameters=model.parameters(),
            **config.optim_hparams)

    loss_record = {
        'train': [],
        'valid': []
    }

    epoch = 0
    early_stop = 0
    min_loss = float('inf')
    timer_train_iter = Timer()
    timer_train = Timer()

    while epoch < n_epochs:
        print(f"epoch {epoch + 1}, "
              f"len train_iter {len(train_iter)}, "
              f"len valid_iter {len(valid_iter)}")

        train_loss, valid_loss = 0.0, 0.0

        timer_train.start()

        model.train()
        for i, (X, Y) in enumerate(train_iter()):
            timer_train_iter.start()
            if 'gpu' in device:
                with paddle.amp.auto_cast():
                    loss = model.loss(model(X), Y)

                scaled = model.scaler.scale(loss)
                scaled.backward()
                model.scaler.minimize(optimizer, scaled)
                optimizer.clear_grad()
                train_loss += model.log_rmse(X, Y)/len(train_iter)
            else:
                loss = model.loss(model(X), Y)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                train_loss += model.log_rmse(X, Y)/len(train_iter)
            timer_train_iter.stop()

        print(f'{timer_train_iter.avg():.5f} sec')
        print(f'{timer_train.stop()/len(train_iter):.5f} sec')

        model.eval()
        for i, (X, Y) in enumerate(valid_iter()):
            with paddle.no_grad():
                loss = model.loss(model(X), Y)
                valid_loss += model.log_rmse(X, Y)/len(valid_iter)

        print(f"train loss {train_loss}, "
              f"valid loss {valid_loss}")

        loss_record['train'].append(train_loss)
        loss_record['valid'].append(valid_loss)

        if valid_loss < min_loss:
            min_loss = valid_loss
            print(f"save model, "
                  f"epoch {epoch + 1}, "
                  f"train loss {loss_record['train'][-1]:f}, "
                  f"valid loss {loss_record['valid'][-1]:f},")
            # del old model
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if f'model_{config.k}' in file:
                        os.remove(os.path.join(root, file))
            # save new model
            paddle.save(model.state_dict(), os.path.join(
                model_dir, f'model_{config.k}_{min_loss:.5f}.pdparams'))
            early_stop = 0
        else:
            early_stop += 1

        if early_stop > n_early_stop:
            break

        epoch += 1
        # if epoch % 10 == 0:
        #     optimizer._learning_rate /= 2.0


if __name__ == '__main__':
    device = paddle.get_device()
    paddle.set_device(device)
    print(f'device {device}')

    config = Config()
    config.device = device

    dataset_dir = 'C:/lbt/ML/datasets/kaggle_california_house_prices.zip'
    train_set, train_labels, _, _ = pre_dataset(dataset_dir)

    for k in range(config.n_k_fold):
        config.k = k
        X_train, y_train, X_valid, y_valid = get_k_fold_data(
            config.n_k_fold, config.k, train_set, train_labels)

        train_dataset = CHP_Dataset(X_train, y_train, mode='train')
        train_iter = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=config.drop_last)

        valid_dataset = CHP_Dataset(X_valid, y_valid, mode='valid')
        valid_iter = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=config.drop_last)

        model = MLP(X_train.shape[1], config)
        train(model, train_iter, valid_iter, config)

    pass
