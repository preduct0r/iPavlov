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
from Task_2_work_ver.Task_2_assignment_2.itmo_kaggle.model import EventDetectionDataset, prepare_shape
from Task_2_work_ver.Task_2_assignment_2.get_seed import set_seed
from torch.utils.data import DataLoader
import pickle
import os
set_seed(43)

pickle_path = r'C:\Users\Andrey\Downloads\itmo_kaggle'
pickle_test_data = pickle.load(open(os.path.join(pickle_path, 'test.pickle'), 'rb'))

i2t = ['Acoustic_guitar',
 'Applause',
 'Bark',
 'Bass_drum',
 'Burping_or_eructation',
 'Bus',
 'Cello',
 'Chime',
 'Clarinet',
 'Computer_keyboard',
 'Cough',
 'Cowbell',
 'Double_bass',
 'Drawer_open_or_close',
 'Electric_piano',
 'Fart',
 'Finger_snapping',
 'Fireworks',
 'Flute',
 'Glockenspiel',
 'Gong',
 'Gunshot_or_gunfire',
 'Harmonica',
 'Hi-hat',
 'Keys_jangling',
 'Knock',
 'Laughter',
 'Meow',
 'Microwave_oven',
 'Oboe',
 'Saxophone',
 'Scissors',
 'Shatter',
 'Snare_drum',
 'Squeak',
 'Tambourine',
 'Tearing',
 'Telephone',
 'Trumpet',
 'Violin_or_fiddle',
 'Writing']

X, names = [], []
for i in pickle_test_data:
    X.append(i['feature'])
    names.append(i['fname'])

test_dset = EventDetectionDataset(X,None,names)

test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=0)

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

model = Transformer(dmodel, output_size, max_len, padding_idx, n_layers,\
                    ffnn_hidden_size, heads, pooling, dropout)

model.load_state_dict(torch.load(r'C:\Users\Andrey\Downloads\itmo_kaggle\model.pb'))

# Move the model to GPU if possible
if CUDA:
    model.cuda()

loss_fn = nn.NLLLoss()

model.add_loss_fn(loss_fn)

device = torch.device('cuda' if CUDA else 'cpu')

model.add_device(device)

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

early_stop = EarlyStopping(wait_epochs=5)

train_losses_list, train_avg_loss_list, train_fscore_list = [], [], []
eval_avg_loss_list, eval_fscore_list, conf_matrix_list = [], [], []
model.eval()
eval_losses = []
losses = []
losses_list = []
fscore = 0
pred_total = torch.LongTensor()
target_total = torch.LongTensor()

df = pd.DataFrame(columns=['fname','label'])
fnames, labels = [], []

with torch.no_grad():
    for i, batch in enumerate(test_loader, 1):
        input_seq, fname = batch
        x_lengths = input_seq.shape[1]

        input_seq.to('cuda')

        preds = model.forward(input_seq.to('cuda'), x_lengths)

        # _, predicted = torch.max(preds.data, 1)

        preds = torch.argmax(preds, 1)

        fnames.append(fname[0])
        labels.append(i2t[int(preds.cpu())])

df['fname'] = fnames
df['label'] = labels
df.to_csv(r'C:\Users\Andrey\Downloads\itmo_kaggle\result.csv', index=False)

