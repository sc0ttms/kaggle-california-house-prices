# -*- coding: utf-8 -*-

import numpy as np
import paddle
import paddle.nn as nn
from config import Config


class MLP(nn.Layer):
    def __init__(self, dim_in, config):
        super().__init__()

        self.dim_embeding = 2048

        self.model = nn.Sequential(
            nn.Linear(dim_in, self.dim_embeding),
            nn.BatchNorm1D(self.dim_embeding),
            nn.ReLU(self.dim_embeding),
            nn.Dropout(config.dropout),

            nn.Linear(self.dim_embeding, 1)
        )

        self.loss = nn.MSELoss()

    def forward(self, X):
        return self.model(X)

    def log_rmse(self, X, Y):
        clipped_preds = paddle.clip(self.model(X), 1, float('inf'))
        eps = np.finfo(np.float32).eps
        log_rmse = paddle.sqrt(
            self.loss(paddle.log(clipped_preds+eps), paddle.log(Y+eps)))
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
    loss = model.loss(model(X), Y)
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
    log_rmse = model.log_rmse(X, Y)

    pass
