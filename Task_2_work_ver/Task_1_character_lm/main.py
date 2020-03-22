import os
from torch.utils.data import DataLoader
from deeppavlov.core.data.simple_vocab import SimpleVocabulary


from Task_2_work_ver.Task_1_character_lm.get_dataset import Dataset, Padder
base_path = r'C:\Users\Andrey'
#==================================================

def read_infile(infile):
    words = []
    with open(infile, encoding='utf-8') as f:
        for line in f.readlines():
            temp = line.split()
            if len(temp)==3:
                words.append(temp[1])
    return words
#==================================================

train_words = read_infile(os.path.join(base_path, "russian-train-high"))
dev_words = read_infile(os.path.join(base_path, "russian-dev"))
test_words = read_infile(os.path.join(base_path, "russian-test"))
print(len(train_words), len(dev_words), len(test_words))
print(*train_words[:10])
#==================================================

vocab = SimpleVocabulary(special_tokens=('PAD', 'UNK', 'BEGIN', 'END'),
                        unk_token='UNK', save_path=r'C:\Users\Andrey\Google Диск\courses\DeepPavlov\Task-2-preduct0r')
vocab.fit([list(x) for x in train_words])
#==================================================

train_dataset = Dataset(train_words, vocab)
dev_dataset = Dataset(dev_words, vocab)
test_dataset = Dataset(test_words, vocab)
#==================================================

train_batcher = DataLoader(train_dataset, batch_size=1)
dev_batcher = DataLoader(dev_dataset, batch_size=1)
test_batcher = DataLoader(test_dataset, batch_size=1)

# for i, (items, classes) in enumerate(train_batcher):
#     print(items.shape, classes.shape)
#     if i==10:
#         break
#==================================================

loader = DataLoader(train_dataset,
                      10,
                      True,
                      collate_fn=Padder(pad_symbol=vocab._t2i['PAD']),
                      pin_memory=True)
#==================================================

for i, (items, classes) in enumerate(loader):
    print(classes)
    break
    if i==10:
        break


