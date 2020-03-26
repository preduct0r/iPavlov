import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset
import torch.nn as nn


class RNNBaseline(nn.Module):
    def __init__(self, input_size, hidden_dim, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(input_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 2)

    def forward(self, seq, hidden=None):
        out = self.emb(seq)
        out, hidden = self.gru(out, hidden)
        out = self.linear(out)
        return out, hidden


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


class Config:
    def __init__(self, lr, batch_size, num_epochs=500):
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
