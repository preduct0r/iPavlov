import os
import pickle
import numpy as np
import torch
import torch.cuda as cuda
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
test_batcher = DataLoader(test_dataset, batch_size=1)

net = torch.load(os.path.join(experiments_path, "net.pb"))


if cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# ============================================================================

# Write a function predict_on_batch that outputs letter probabilities of all words in the batch.
h = net.init_hidden(1)

def generate(model, max_length=20, start_index=2, end_index=3, h=h, device=device):
    cur_index, length = start_index, 0
    word = ''
    while length!=max_length:
        outputs, h = model(torch.LongTensor([[cur_index]]).to(device), h)
        outputs = softmax(outputs, dim=2)
        outputs = outputs.view(-1, len(vocab))
        cur_index = torch.multinomial(outputs.squeeze(), 1).detach().cpu().numpy()[0]
        length += 1
        if cur_index==end_index:
            break
        word += vocab._i2t[cur_index]

    print(word)
generate(net)

