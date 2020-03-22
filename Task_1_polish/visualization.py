import bokeh.models as bm, bokeh.plotting as pl
from bokeh.io import output_notebook
import matplotlib.pyplot as plt
import seaborn as sns
# from tsnecuda import TSNE
from sklearn.manifold import TSNE
from sklearn import preprocessing
import torch
from torch import nn
import os
import pickle
from matplotlib import pylab


data_path = r"C:\Users\Andrey\Google Диск\courses\DeepPavlov\data\Task_1"

with open(os.path.join(data_path, 'w2v_data.pkl'), "rb") as f:
   [X, y, word2index, index2word] = pickle.load(f)

class model(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
      super().__init__()

      self.emb = nn.Embedding(vocab_size, embedding_dim)
      self.linear = nn.Linear(embedding_dim, vocab_size)

net = model(len(word2index.keys()), 300)

net = torch.load(os.path.join(data_path, 'net_text8.pb'))

weights = net.emb.weight.detach().cpu().numpy()

num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(weights[1:num_points+1, :])

def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    pylab.scatter(x, y)
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  pylab.show()

words = [index2word[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)