import torch
from torch import cuda
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import numpy as np


class My_Dataset(Dataset):
    def __init__(self, x, y):
        if cuda.is_available():
            self.x = torch.cuda.LongTensor(x)
            self.y = torch.cuda.LongTensor(y)

        else:
            self.x = torch.LongTensor(x)
            self.y = torch.LongTensor(y)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx]


class EarlyStopping():
    def __init__(self, patience=5, min_percent_gain=0.1):
        self.patience = patience
        self.loss_list = []
        self.min_percent_gain = min_percent_gain / 100.

    def update_loss(self, loss):
        self.loss_list.append(loss)
        if len(self.loss_list) > self.patience:
            del self.loss_list[0]

    def stop_training(self):
        if len(self.loss_list) == 1:
            return False
        gain = (max(self.loss_list) - min(self.loss_list)) / max(self.loss_list)
        print("Loss gain: {}%".format(round(100 * gain, 2)))
        if gain < self.min_percent_gain:
            return True
        else:
            return False


def weights_init(m):
    classname = m.__class__.__name__

    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0, 1 / np.sqrt(y))
        # m.bias.data should be 0
        m.bias.data.fill_(0)

    elif classname.find('Embedding') != -1:
        torch.nn.init.xavier_uniform_(m.weight)


class model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, word2index):
      super().__init__()

      self.emb = nn.Embedding(vocab_size, embedding_dim)
      self.linear = nn.Linear(embedding_dim, vocab_size)
      self.word2index = word2index


    def forward(self, x):
      out = torch.sum(self.emb(x), dim=1)  # [batch_size, 2*window_size, 20000] => [batch_size, 20000]
      out = self.linear(out)
      # print(out.shape)
      out = F.log_softmax(out, dim=1)
      # print(out.shape)

      return out

    def word_embeddings(self, word):                          # функция для п.3 function to map token to corresponding word vector
      word = torch.LongTensor([self.word2index[word]])
      if cuda.is_available():
          word = word.cuda()
      else:
          self.cpu()
      return self.emb(word)

def get_batcher(X, y, batch_size):
    return DataLoader(My_Dataset(X, y), batch_size=batch_size, shuffle=False)

