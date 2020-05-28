from Task_2_work_ver.Task_2_assignment_2.huawei.early_stopping import EarlyStopping
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

import torch.nn.functional as F
from torch import device
from tqdm import tqdm_notebook
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from Task_2_work_ver.Task_2_assignment_2.itmo_kaggle.itmo_transformer import Transformer, LabelSmoothingLoss
from Task_2_work_ver.Task_2_assignment_2.itmo_kaggle.model import EventDetectionDataset, \
    prepare_shape, torch_model, DummyNetwork, BaseLineModel
from Task_2_work_ver.Task_2_assignment_2.get_seed import set_seed
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
import pickle
import os
set_seed(43)

pickle_path = r'C:\Users\Andrey\Downloads\itmo_kaggle'
pickle_train_data = pickle.load(open(os.path.join(pickle_path, 'train.pickle'), 'rb'))
pickle_test_data = pickle.load(open(os.path.join(pickle_path, 'test.pickle'), 'rb'))

X, y = [], []
for i in pickle_train_data:
    X.append(i['feature'])
    y.append(i['label_id'])

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2)

train_dset = EventDetectionDataset(X_train, y_train)
val_dset = EventDetectionDataset(X_val, y_val)

train_loader = DataLoader(train_dset, batch_size=400, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dset, batch_size=400, shuffle=False, num_workers=0)

batch_size = 400
dmodel = 128
max_len = 250
output_size = 41
padding_idx = 0
n_layers = 4
ffnn_hidden_size = dmodel * 2
heads = 8
pooling = 'avg'
dropout = 0.5
label_smoothing = False #0.1
learning_rate = 0.00001
epochs = 100

# Check whether system supports CUDA
CUDA = torch.cuda.is_available()

# model = Transformer(dmodel, output_size, max_len, padding_idx, n_layers,\
#                     ffnn_hidden_size, heads, pooling, dropout)

model = DummyNetwork()
model.to('cuda')

# Move the model to GPU if possible
if CUDA:
    model.cuda()

# Add loss function
if label_smoothing:
    loss_fn = LabelSmoothingLoss(output_size, label_smoothing)
else:
    loss_fn = nn.NLLLoss()

# model.add_loss_fn(loss_fn)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# model.add_optimizer(optimizer)

device = torch.device('cuda' if CUDA else 'cpu')

# model.add_device(device)

# Create the parameters dictionary and instantiate the tensorboardX SummaryWriter
params = {'batch_size': batch_size,
          'dmodel': dmodel,
          'n_layers': n_layers,
          'ffnn_hidden_size': ffnn_hidden_size,
          'heads': heads,
          'pooling': pooling,
          'dropout': dropout,
          'label_smoothing': label_smoothing,
          'learning_rate': learning_rate}
# train_writer = SummaryWriter(comment=f' Training, batch_size={batch_size}, dmodel={dmodel}, n_layers={n_layers},\
# ffnn_hidden_size={ffnn_hidden_size}, heads={heads}, pooling={pooling}, dropout={dropout}, \
# label_smoothing={label_smoothing}, learning_rate={learning_rate}'.format(**params))
#
# val_writer = SummaryWriter(comment=f' Validation, batch_size={batch_size}, dmodel={dmodel}, n_layers={n_layers},\
# ffnn_hidden_size={ffnn_hidden_size}, heads={heads}, pooling={pooling}, dropout={dropout}, \
# label_smoothing={label_smoothing}, learning_rate={learning_rate}'.format(**params))

# Instantiate the EarlyStopping
# early_stop = EarlyStopping(wait_epochs=5)
#
# train_losses_list, train_avg_loss_list, train_fscore_list = [], [], []
# eval_avg_loss_list, eval_fscore_list, conf_matrix_list = [], [], []
# min_loss = 1000

# for epoch in range(epochs):
#
#     print('\nStart epoch [{}/{}]'.format(epoch + 1, epochs))
#
#     train_losses, train_avg_loss, train_fscore = model.train_model(train_loader)
#
#     train_losses_list.append(train_losses)
#     train_avg_loss_list.append(train_avg_loss)
#     train_fscore_list.append(train_fscore)
#
#     _, eval_avg_loss, eval_fscore = model.evaluate_model(val_loader)
#
#     eval_avg_loss_list.append(eval_avg_loss)
#     eval_fscore_list.append(eval_fscore)
#     # conf_matrix_list.append(conf_matrix)
#
#     if eval_avg_loss < min_loss:
#         torch.save(model.state_dict(), r'C:\Users\Andrey\Downloads\itmo_kaggle\model.pb')
#         min_loss = eval_avg_loss
#
#     print(
#         '\nEpoch [{}/{}]: Train fscore: {:.3f}. Train loss: {:.4f}. Evaluation fscore: {:.3f}. Evaluation loss: {:.4f}' \
#         .format(epoch + 1, epochs, train_fscore, train_avg_loss, eval_fscore, eval_avg_loss))
#
#     # if early_stop.stop(eval_avg_loss, model, delta=0.003):
#     #     break
#
# with open(r'C:\Users\Andrey\Downloads\itmo_kaggle\data_from_training.pkl', 'wb') as f:
#     pickle.dump([train_avg_loss_list, train_fscore_list, eval_avg_loss_list, eval_fscore_list], f)


n_epoch = 30
criterion = nn.NLLLoss()
network = DummyNetwork()
# можно попробовать другой optimizer, тоже считается улучшением
optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
network.to('cuda')


train_loss = []
val_loss = []

train_acc = []
val_acc = []
min_loss = 1000

for e in range(n_epoch):
    print('epoch #', e)
    # train
    loss_list = []
    outputs = []
    targets = []
    for i_batch, sample_batched in enumerate(train_loader):
        x, y = sample_batched
        optimizer.zero_grad()

        output = network(x.to('cuda'))
        outputs.append(output.cpu().detach().numpy().argmax(axis=1))

        target = y
        targets.append(target)

        zzz = torch.LongTensor(outputs[-1])
        zzz2 = target.long()
        loss = criterion(output.cpu(), zzz2)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
    y_true = np.hstack(targets)
    y_pred = np.hstack(outputs)
    acc = f1_score(y_true, y_pred, average='macro')
    train_loss.append(np.mean(loss_list))
    train_acc.append(acc)
    print('mean train loss:', train_loss[-1])
    print('train fscore:', acc)

    loss_list = []
    outputs = []
    targets = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(val_loader):
            x, y = sample_batched
            optimizer.zero_grad()

            output = network(x.to('cuda'))
            outputs.append(output.cpu().detach().numpy().argmax(axis=1))

            target = y
            targets.append(target)

            loss = criterion(output.cpu(), target.long())
            loss_list.append(loss.item())
        #             loss.backward()
        #             optimizer.step()

        y_true = np.hstack(targets)
        y_pred = np.hstack(outputs)
        acc = f1_score(y_true, y_pred, average='macro')
        val_loss.append(np.mean(loss_list))
        val_acc.append(acc)
        print('mean val loss:', val_loss[-1])
        print('val fscore:', acc)

        if val_loss[-1] < min_loss:
            torch.save(model.state_dict(), r'C:\Users\Andrey\Downloads\itmo_kaggle\dummy_model.pb')
            min_loss = val_loss[-1]


