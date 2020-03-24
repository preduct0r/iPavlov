import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset
import torch.nn as nn


def get_rid_of_pad(outputs, classes, vocab):
    indexes_for_loss = (classes==vocab._t2i['PAD']).nonzero()
    classes_for_loss = classes[indexes_for_loss].view(-1)
    outputs_for_loss = outputs[indexes_for_loss, :].view(-1,len(vocab))
    return outputs_for_loss, classes_for_loss


def read_infile(infile):
    words = []
    with open(infile, encoding='utf-8') as f:
        for line in f.readlines():
            temp = line.split()
            if len(temp)==3:
                words.append(temp[1])
    return words


class Dataset(TorchDataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __getitem__(self, index):
        """
        Returns one tensor pair (source and target). The source tensor corresponds to the input word,
        with "BEGIN" and "END" symbols attached. The target tensor should contain the answers
        for the language model that obtain these word as input.
        """
        word = self.data[index]
        batch = self.vocab.__call__(['BEGIN'] + list(word) + ['END'])
        return (torch.LongTensor(batch[:-1]), torch.LongTensor(batch[1:]))

    def __len__(self):
        return len(self.data)
#==================================================


def pad_tensor(vec, length, dim, pad_symbol):
    """
    Pads a vector ``vec`` up to length ``length`` along axis ``dim`` with pad symbol ``pad_symbol``.
    """
    out_dims = list(vec.shape)
    pad = np.zeros((len(out_dims)*2,)).astype(int).tolist()
    k = len(out_dims)-dim-1
    pad[2*k+1] = length-out_dims[dim]
    return torch.nn.functional.pad(vec, pad, mode='constant', value=pad_symbol)

class Padder:
    def __init__(self, dim=0, pad_symbol=0):
        self.dim = dim
        self.pad_symbol = pad_symbol

    def __call__(self, batch):
        lengths = sorted([t[0].shape[self.dim] for t in batch])
        data = torch.stack([pad_tensor(t[0], lengths[-1], self.dim, self.pad_symbol) for t in batch])
        targets = torch.stack([pad_tensor(t[1], lengths[-1], self.dim, self.pad_symbol) for t in batch])
        return [data, targets]
#==================================================


class RNNLM(nn.Module):
    def __init__(self, vocab_size, embeddings_dim, hidden_size, batch_size, device):
        super(RNNLM, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embeddings_dim = embeddings_dim
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, embeddings_dim)
        self.gru = nn.GRU(embeddings_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, inputs, hidden):
        out = self.emb(inputs)
        out, hidden = self.gru(out)
        out = self.linear(out)
        # out = self.softmax(out)
        return out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = list(self.parameters())[1].data                                              # ???
        # weight = next(self.parameters()).data
        hidden = (weight.new_tensor((1, batch_size, self.hidden_size)).zero_().to(self.device),
                      weight.new_tensor((1, batch_size, self.hidden_size)).zero_().to(self.device))
        return hidden


def validate_on_batch(model, criterion, x, y, h, vocab):
    outputs, h = model(x, h).view(-1,len(vocab))
    return outputs, h, criterion(outputs, y)


def train_on_batch(model, criterion, x, y, optimizer, iter_loss, vocab):
    optimizer.zero_grad()
    outputs, loss = validate_on_batch(model, criterion, x, y, vocab)
    loss.backward()
    iter_loss += loss.item()
    optimizer.step()
    return outputs, iter_loss


def predict_on_batch(model, criterion, batch):
    pass

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




