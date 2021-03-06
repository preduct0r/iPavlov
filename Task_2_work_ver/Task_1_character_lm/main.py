import warnings
warnings.simplefilter(action='ignore')
import os
import time
import pickle
import torch
from torch.nn.functional import log_softmax
import torch.cuda as cuda
from torch.utils.data import DataLoader
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from Task_2_work_ver.Task_1_character_lm.get_func import Config, Dataset, Padder, RNNLM, EarlyStopping, \
                                      validate_on_batch, train_on_batch, read_infile, get_rid_of_pad

base_path = r'C:\Users\Andrey'
experiments_path = r'C:\Users\Andrey\Google Диск\courses\DeepPavlov\Task-2-preduct0r\data\Task_1'
config = Config(lr=0.0001, batch_size=128, num_epochs=300)

train_words = read_infile(os.path.join(base_path, "russian-train-high"))
dev_words = read_infile(os.path.join(base_path, "russian-dev"))
test_words = read_infile(os.path.join(base_path, "russian-test"))

vocab = SimpleVocabulary(special_tokens=('PAD', 'UNK', 'BEGIN', 'END'),
                        unk_token='UNK', save_path=experiments_path)
vocab.fit([list(x) for x in train_words])

train_dataset = Dataset(train_words, vocab)
dev_dataset = Dataset(dev_words, vocab)
test_dataset = Dataset(test_words, vocab)
# ==========================================================================

if cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device='cpu'

net = RNNLM(vocab_size=len(vocab), embeddings_dim=len(vocab), hidden_size=300, batch_size=config.batch_size, device=device)
net.to(device)

train_batcher = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=Padder(pad_symbol=vocab._t2i['PAD']))
dev_batcher = DataLoader(dev_dataset, batch_size=config.batch_size, collate_fn=Padder(pad_symbol=vocab._t2i['PAD']))
test_batcher = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=Padder(pad_symbol=vocab._t2i['PAD']))

start_time = time.time()

train_loss = []
valid_loss = []
train_fscore = []
valid_fscore = []
min_loss =1000

weights = torch.ones((len(vocab),)).to(device)
weights[vocab._t2i['PAD']]  = 0
criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
net.to(device)

early_stopping = EarlyStopping()

for epoch in range(config.num_epochs):
    # h = net.init_hidden(config.batch_size)
    iter_loss = 0.0
    correct = 0
    iterations = 0
    net.train()

    for i, (items, classes) in enumerate(train_batcher):
        optimizer.zero_grad()

        items = items.to(device)
        classes = classes.to(device).view(-1)

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        # h = [each.data for each in h]                                      #???

        outputs, hidden = net(items)
        outputs = outputs.view(-1, len(vocab))
        # outputs = log_softmax(outputs.view(-1, len(vocab)), dim=1)

        loss = criterion(outputs, classes)
        loss.backward()
        iter_loss += loss.item()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == classes.data.long()).sum()

        iterations += 1

        torch.cuda.empty_cache()

    train_loss.append(iter_loss / iterations)

    # early_stopping.update_loss(train_loss[-1])
    # if early_stopping.stop_training():
    #     break

    ############################
    # Validate
    ############################
    # val_h = net.init_hidden(config.batch_size)
    iter_loss = 0.0
    correct = 0
    f_scores = 0
    iterations = 0
    net.eval()

    for i, (items, classes) in enumerate(dev_batcher):
        items = items.to(device)
        classes = classes.to(device).view(-1)

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        # val_h = [each.data for each in val_h]

        outputs, val_h = net(items)
        outputs = outputs.view(-1, len(vocab))
        # outputs = log_softmax(outputs.view(-1, len(vocab)), dim=1)
        loss = criterion(outputs, classes)
        iter_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == classes.data.long()).sum()

        iterations += 1

    valid_loss.append(iter_loss / iterations)

    if epoch%10==0 and epoch!=0 and valid_loss[-1] < min_loss:
        torch.save(net, os.path.join(experiments_path, "net.pb"))
        min_loss = valid_loss[-1]



    print('Epoch %d/%d, Tr Loss: %.4f, Dev Loss: %.4f'
          % (epoch + 1, config.num_epochs, train_loss[-1], valid_loss[-1]))

with open(os.path.join(experiments_path, "loss_track.pkl"), 'wb') as f:
    pickle.dump([train_loss, valid_loss], f)
