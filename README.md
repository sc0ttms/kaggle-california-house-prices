# README

![Platform](https://img.shields.io/badge/Platform-win10--64-lightgrey)
![Python](https://img.shields.io/badge/Python-3.7.5-orange)
![PowerShell](https://img.shields.io/badge/PowerShell-7.1.5-orange)
![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-2.1.3-orange)

## 项目简介

动手学深度学习房价预测练习

kaggle竞赛网址 [kaggle california house prices](https://www.kaggle.com/c/california-house-prices)

## 代码分析

### 配置文件`config.py`

- device: 使用`cpu`还是`gpu`训练
- batch_size: mini-batch大小
- optimizer：优化算法使用`Adam`
- optim_hparams: 优化算法超参数
  - learning_rate: 学习率
  - weight_decay: 权重衰退
- num_workers: `Dataloader`线程数
- drop_last: 是否丢弃不够mini-batch大小的数据
- dropout: 丢弃比例
- n_epochs: epoch总数
- n_early_stop: loss不下降时等多少个epoch停止训练
- n_k_fold: k折交叉验证个数
- k: k折交叉验证当前第几个
  
### 数据处理`dataset.py`

数据处理分三个部分: 数据预处理/k折交叉取数据/数据加载

> 数据预处理

- 将`object`的数据去除掉
- 对数据进行均值为0，方差为1的标准化
- 将`nan`数据替换为0
- 将数据分为`train_set/train_labels/test_set/test_id`
  - `test_id`主要用于test时填充test_id列

> k折交叉取数据

- 将`train_set/train_labels`分成k份
- 取其中一份作为`valid_set/valid_labels`，其余部分作为真正的`train_set/train_labels`

> 数据加载

- 使用`CHP_Dataset`和`DataLoader`加载mini-batch大小的数据

### 模型定义`model.py`

- 定义单隐藏层`MLP`
- 定义损失函数为`MSELoss`
- 定义`log_rmse`函数，为了与kaggle结果评估方式一致

### 训练`train.py`

k折交叉验证训练，最终得到k个模型

### 推理`test.py`

选择k个模型中损失最小的模型进行推理

## License

[![License](https://img.shields.io/badge/License-BSD-green)](https://github.com/sc0ttms/kaggle-california-house-prices/blob/main/LICENSE)
