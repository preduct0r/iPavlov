import torch
from torch import cuda
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle

from Task_1_work_ver.data_processing import Data_Processing_CBOW
from Task_1_work_ver.my_model import model, get_batcher, EarlyStopping


with open(r"C:\Users\Andrey\Google Диск\DeepPavlov\text8\text8","r") as f:
  data = f.read()

intermediate_results_path = r"C:\Users\Andrey\PycharmProjects\DeepPavlov\intermediate_results"

# X, y, word2index, index2word = Data_Processing_CBOW(data, 5, intermediate_results_path).get_data()
with open(os.path.join(intermediate_results_path, "w2v_data.pkl"), "rb") as f:
  [X, y, word2index, index2word] = pickle.load(f)

net = model(len(word2index.keys()), 300, word2index)

print_every = 10
num_epochs = 100
batch_size = 512
lr = 0.01

optimizer = optim.SGD(net.parameters(), lr=0.01)

batcher_train = get_batcher(X,y,batch_size)

# https://github.com/jeffchy/pytorch-word-embedding/blob/master/CBOW.py              самая толковая версия
# https://github.com/FraLotito/pytorch-continuous-bag-of-words/blob/master/cbow.py   реализована get_embeddings

if cuda.is_available():
  net = net.cuda()

# cбросить веса
# net.apply(weights_init)

train_loss = []
train_accuracy = []

early_stopping = EarlyStopping()

for epoch in range(num_epochs):

  iter_loss = 0.0
  correct = 0
  iterations = 0

  net.train()  # Put the network into training mode

  for i, (items, classes) in enumerate(batcher_train):

    # Convert torch tensor to Variable
    items = Variable(items)
    classes = Variable(classes)

    # If we have GPU, shift the data to GPU
    # if cuda.is_available():
    #   items = items.cuda()
    #   classes = classes.cuda()

    net.zero_grad()  # Clear off the gradients from any past operation
    outputs = net(items)  # Do the forward pass

    lr = lr / (2 ** (epoch // 10))
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr

    loss = F.nll_loss(outputs, classes)
    iter_loss += loss.item()  # Accumulate the loss
    loss.backward()  # Calculate the gradients with help of back propagation
    optimizer.step()  # Ask the optimizer to adjust the parameters based on the gradients

    # Record the correct predictions for training data
    _, predicted = torch.max(outputs.data, 1)

    correct += (predicted == classes.data).sum()
    iterations += 1

  # Record the training loss
  train_loss.append(iter_loss / iterations)
  # Record the training accuracy
  train_accuracy.append((100 * correct / len(batcher_train.dataset)))
  print(100.0 * correct / len(batcher_train.dataset))

  train_loss.append(train_loss[-1])
  train_accuracy.append(train_accuracy[-1])

  early_stopping.update_loss(train_loss[-1])
  if early_stopping.stop_training():
    break

  # if epoch%print_every==0:
  print('Epoch %d/%d, Tr Loss: %.4f, Tr Accuracy: %.1f'
        % (epoch + 1, num_epochs, train_loss[-1], train_accuracy[-1]))

torch.save(net, os.path.join(intermediate_results_path, "net.pb"))
