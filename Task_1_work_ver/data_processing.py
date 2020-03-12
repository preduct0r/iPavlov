import numpy as np
from collections import Counter
import pickle
import os
from string import punctuation


# считаем слова и сортируем по убыванию количества вхождений
def counter(words):
    count_words = Counter(words)
    total_words = len(words)
    sorted_words = count_words.most_common(total_words)
    print('total words: {}'.format(total_words))
    print(sorted_words[150:200])
    less_10, more_10 = 0, 0
    for word, freq in sorted_words:
        if freq > 10:
            more_10 += 1
        if freq < 10:
            less_10 += 1
    print('more then 10: {}, less then 10: {}'.format(more_10, less_10))
    return sorted_words, more_10

# назначаю каждому токену номер
def make_dicts(sorted_words, dimensionality):
    word2index = {w: i + 1 for i, (w, c) in enumerate(sorted_words[:(dimensionality - 1)])}
    word2index['UNK'] = 0
    index2word = {i + 1: w for i, (w, c) in enumerate(sorted_words[:(dimensionality - 1)])}
    index2word[0] = 'UNK'
    return word2index, index2word


class Data_Processing_CBOW():
    def __init__(self, data, window_size, path):
        self.window_size = window_size
        self.dimensionality = np.nan
        self.path = path
        self.corpus = ''.join([c.lower() for c in data if c not in punctuation + '«»']).split()

    def get_data(self):
        sorted_words, self.dimensionality = counter(self.corpus)
        word2index, index2word = make_dicts(sorted_words, self.dimensionality)
        # собственно ооздаемем X,y
        X,y = [],[]
        num_sent = [word2index[x] if x in word2index.keys() else 0 for x in self.corpus]
        for cur_idx,num in enumerate(num_sent[self.window_size:-self.window_size]):
          idx = cur_idx+self.window_size
          y.append(num)
          X.append(num_sent[idx-self.window_size:idx]+num_sent[idx+1:idx+self.window_size+1])

        with open(os.path.join(self.path, "w2v_data.pkl"), "wb") as f:
            pickle.dump([np.array(X), np.array(y), word2index, index2word], f)

        return np.array(X), np.array(y), word2index, index2word