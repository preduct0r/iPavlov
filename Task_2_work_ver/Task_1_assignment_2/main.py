import os
import dill
import pandas as pd
import numpy as np
import torch
import torch.cuda as cuda
from torchtext import datasets
from torchtext.data import Field, LabelField
from torchtext.data import BucketIterator
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Task_2_work_ver.Task_1_assignment_2.loader import saver, loader
from Task_2_work_ver.Task_1_assignment_2.get_func import RNNBaseline, Config, EarlyStopping


data_path = r'C:\Users\Andrey\Google Диск\courses\DeepPavlov\Task-2-preduct0r\data\task3'
# TEXT = Field(sequential=True, lower=True)
# LABEL = LabelField()
#
# train, tst = datasets.IMDB.splits(TEXT, LABEL)
# trn, vld = train.split()
#
# TEXT.build_vocab(trn)
# LABEL.build_vocab(trn)
# #
# saver([TEXT, LABEL], 'TEXT_LABEL')

[TEXT, LABEL] = loader('TEXT_LABEL')

train, tst = datasets.IMDB.splits(TEXT, LABEL)
trn, vld = train.split()

# zz = [next(tst.label.__iter__()) for i in range(len(tst))]
# zzz = np.unique(zz, return_counts=True)


if cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device='cpu'

train_iter, val_iter, test_iter = BucketIterator.splits(
        (trn, vld, tst),
        batch_sizes=(64, 64, 64),
        sort=False,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        device='cuda',
        repeat=False
)

batch = next(train_iter.__iter__())
print(batch.__dict__.keys())
print(batch.text.shape)
print(batch.label.shape)
print(batch.batch_size)
print(batch.dataset)
print(batch.fields)
# ======================================================
# попробуй реализовать Plateou func

config = Config(lr=10e-5, batch_size=128, num_epochs=500)
em_sz = 200
nh = 300
net = RNNBaseline(TEXT.  nh, em_sz)

net.to(device)

opt = optim.Adam(net.parameters(), lr=config.lr)
loss_func = nn.CrossEntropyLoss()

for epoch in range(1, config.num_epochs + 1):
    iter = 0
    running_loss = 0.0
    running_corrects = 0
    net.train()
    for batch in train_iter:
        x = batch.text
        y = batch.label

        opt.zero_grad()
        preds = net(x)
        loss = loss_func(preds, y)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        iter+=1

    epoch_loss = running_loss / iter

    iter = 0
    val_loss = 0.0
    net.eval()
    for batch in val_iter:
        x = batch.text
        y = batch.label

        preds = net(x)
        loss = loss_func(preds, y)
        val_loss += loss.item()
        iter+=1

    val_loss /= iter
    print('Epoch: {}, Training Loss: {}, Validation Loss: {}'.format(epoch, epoch_loss, val_loss))
