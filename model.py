# -*- coding: utf-8 -*-

import paddle
import paddle.nn as nn
from config import Config


class MLP(nn.Layer):
    def __init__(self, dim_in, config):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.BatchNorm1D(dim_in),
            nn.ReLU(dim_in),
            nn.Dropout(config.dropout),

            nn.Linear(dim_in, dim_in),
            nn.BatchNorm1D(dim_in),
            nn.ReLU(dim_in),
            nn.Dropout(config.dropout),

            nn.Linear(dim_in, 1)
        )

        self.loss = nn.MSELoss()

        if 'gpu' in config.device:
            self.scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    def forward(self, X):
        return self.model(X)

    def log_rmse(self, X, Y):
        clipped_preds = paddle.clip(self.model(X), 1, float('inf'))
        log_rmse = paddle.sqrt(
            self.loss(paddle.log(clipped_preds), paddle.log(Y)))
        return log_rmse.item()


if __name__ == '__main__':
    device = paddle.get_device()
    paddle.set_device(device)
    print(f'device {device}')

    config = Config()
    config.device = device

    X = paddle.randn([3, 18])
    Y = paddle.randn([3, 1])

    model = MLP(X.shape[1], config)
    print(model)

    optimizer = getattr(
        paddle.optimizer, config.optimizer)(
            parameters=model.parameters(),
            **config.optim_hparams)
    print(optimizer)

    log_rmse = 0.0
    if 'gpu' in config.device:
        with paddle.amp.auto_cast():
            loss = model.loss(model(X), Y)
        scaled = model.scaler.scale(loss)
        scaled.backward()
        model.scaler.minimize(optimizer, scaled)
        optimizer.clear_grad()
        log_rmse = model.log_rmse(X, Y)
    else:
        loss = model.loss(model(X), Y)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

    pass
