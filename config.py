# -*- coding: utf-8 -*-

class Config():
    def __init__(self):
        self.device = 'cpu'

        self.batch_size = 512
        self.optimizer = 'Adam'
        self.optim_hparams = {
            'learning_rate': 0.05,
            'weight_decay': None
        }

        self.num_workers = 2
        self.drop_last = False

        self.dropout = 0.1

        self.n_epochs = 5000
        self.n_early_stop = 200

        self.n_k_fold = 5
        self.k = 0
        pass
