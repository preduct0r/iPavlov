from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
import tqdm
import pprint
import os
import pickle
import torch
from torch import cuda
from torch import nn

data_path = r"C:\Users\Andrey\Google Диск\courses\DeepPavlov\data\Task_1"

with open(os.path.join(data_path, 'w2v_data.pkl'), "rb") as f:
   [X, y, word2index, index2word] = pickle.load(f)

class model(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
      super().__init__()

      self.emb = nn.Embedding(vocab_size, embedding_dim)
      self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        out = torch.sum(self.emb(x), dim=1)  # [batch_size, 2*window_size, 20000] => [batch_size, 20000]
        out = self.linear(out)
        # print(out.shape)
        out = F.log_softmax(out, dim=1)
        # print(out.shape)

        return out

    def word_embeddings(self, word):  # функция для п.3 function to map token to corresponding word vector
        word = torch.LongTensor([word2index[word]])
        if cuda.is_available():
            word = word.cuda()
        else:
            self.cpu()
        return self.emb(word)

net = model(len(word2index.keys()), 300)

net = torch.load(os.path.join(data_path, 'net_text8.pb'))

weights = net.emb.weight.detach().cpu().numpy()

# embedding = WordEmbeddingsKeyedVectors(vector_size=300)
# for i, n in enumerate(word2index.keys()):
#     embedding.add(entities=n, weights=net.word_embeddings(n).cpu().detach())
#     if not i % 100:
#         print(f'{i}, {n}')
#
# embedding.save(os.path.join(data_path, 'keyed_values.dir'))

# =====================================================================================
def analogy(x1, x2, y1):
    result = embedding.most_similar(positive=[y1, x2], negative=[x1])
    return result[0][0]

embedding = WordEmbeddingsKeyedVectors.load(os.path.join(data_path, 'keyed_values.dir'))
print(analogy('estimate', 'estimates', 'find'))

accuracy, result = embedding.evaluate_word_analogies(os.path.join(data_path, 'intrinsic_test.txt'))
print(accuracy)
for r in result:
    correct_len = len(r['correct'])
    incorrect_len = len(r['incorrect'])
    print(f'{r["section"]}: {correct_len} / {(correct_len + incorrect_len)}')

# =====================================================================================

from gensim.test.utils import datapath

print(
    (embedding.n_similarity(["king"], ["duke"]),
     embedding.n_similarity(["king"], ["queen"]),
     embedding.most_similar(positive=['woman', 'king'], negative=['man']),
     embedding.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant']),))

embedding.evaluate_word_pairs(datapath('wordsim353.tsv'))