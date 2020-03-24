import os
import pickle
import numpy as np
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from Task_2_work_ver.Task_1_character_lm.plot_loss import plot_loss


from Task_2_work_ver.Task_1_character_lm.get_func import read_infile, Dataset, Padder, Config


base_path = r'C:\Users\Andrey'
experiments_path = r'C:\Users\Andrey\Google Диск\courses\DeepPavlov\Task-2-preduct0r\data\Task_1'

train_words = read_infile(os.path.join(base_path, "russian-train-high"))
test_words = read_infile(os.path.join(base_path, "russian-test"))
vocab = SimpleVocabulary(special_tokens=('PAD', 'UNK', 'BEGIN', 'END'),
                        unk_token='UNK', save_path=experiments_path)
vocab.fit([list(x) for x in train_words])
config = Config(lr=0.0001, batch_size=512, num_epochs=1000)
test_dataset = Dataset(test_words, vocab)
test_batcher = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=Padder(pad_symbol=vocab._t2i['PAD']))

net = torch.load(os.path.join(experiments_path, "net.pb"))
# ============================================================================


# Write a function predict_on_batch that outputs letter probabilities of all words in the batch.
h = net.init_hidden(1)
def predict_on_batch(model, criterion, x, y, vocab, h=h):
    # detach hidden state from history
    h = tuple([each.data for each in h])

    outputs, h = model(x, h)
    outputs = softmax(outputs, dim=2)
    outputs = outputs.view(-1,len(vocab))
    letters = [vocab._i2t[i] for i in y]
    preds = outputs[np.arange(y.shape[0]), y]
    print([vocab._i2t[s] for s in np.argmax(outputs.detach().cpu().numpy(), axis=1)])
    result = ''
    for i, prob in zip(letters,preds.detach().cpu().numpy()):
        result += '{}({})'.format(i, str(np.round_(prob, decimals=3)))
    return result


loss = 0.0
correct = 0
iterations = 0
f_scores = 0

criterion = torch.nn.CrossEntropyLoss(ignore_index = vocab._t2i['PAD'])
net.eval()
torch.manual_seed(7)

for i, (items, classes) in enumerate(test_batcher):
    items = items.to('cuda')
    classes = classes.to('cuda').view(-1)
    print(predict_on_batch(net, criterion, items, classes, vocab))

    outputs = net(items).view(-1,len(vocab))
    loss += criterion(outputs, classes.long()).item()

    _, predicted = torch.max(outputs.data, 1)
    correct += (predicted == classes.data.long()).sum()

    iterations += 1

print(loss / iterations)


