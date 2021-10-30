# -*- coding: utf-8 -*-

class Config():
    def __init__(self):
        self.device = 'cpu'

        self.batch_size = 512
        self.optimizer = 'Adam'
        self.optim_hparams = {
            'learning_rate': 1e-2,
            'weight_decay': 1e-5
        }

        self.num_workers = 0
        self.drop_last = True

        self.dropout = 0.1

        self.n_epochs = 5000
        self.n_early_stop = 200

        self.n_k_fold = 5
        self.k = 0
        pass
