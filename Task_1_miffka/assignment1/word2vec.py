import argparse
import os.path as osp

import numpy as np
from scipy.spatial.distance import cdist

from Task_1_miffka.assignment1.config import config

class Word2Vec(object):
    def __init__(self, vectors_path, meta_path):
        self.vectors = np.loadtxt(vectors_path, delimiter='\t')
        self.word2ind = {token: number for number, token in enumerate(np.loadtxt(meta_path, dtype=object))}
        self.ind2word = {number: token for token, number in self.word2ind.items()}
    
    def word2index(self, word):
        if word in self.word2ind:
            idx = self.word2ind.get(word)
        else:
            print(f'[WARNING] Word {word} not found in dict, return embedding for "UNK"')
            idx = self.word2ind.get('UNK')
        return idx
    
    def index2word(self, idx):
        return self.ind2word[idx]
    
    def __getitem__(self, idx):
        return self.vectors[self.word2index(idx)]
    
    def most_similar(self, positive, negative=None, topn=10):
        if type(positive) not in {list, np.ndarray}:
            positive = [positive]
        if negative is not None:
            if type(negative) not in {list, np.ndarray}:
                negative = [negative]
        else:
            negative = []
        pos_ids = [self.word2index(pos_word) for pos_word in positive]
        neg_ids = [self.word2index(neg_word) for neg_word in negative]
        final_vector = (np.sum([self.vectors[pos_id] for pos_id in pos_ids], axis=0) - \
                       np.sum([self.vectors[neg_id] for neg_id in neg_ids], axis=0)).reshape(1, -1)
        dist_vector = cdist(self.vectors, final_vector, metric='cosine')
        ids = np.argsort(dist_vector, axis=0)
        ids = np.asarray([ind for ind in ids if ind not in pos_ids + neg_ids][:topn], dtype=np.int).flatten()
        sims = [1 - dist_vector[ind][0] for ind in ids]
        return [(self.index2word(idx), sim) for idx, sim in zip(ids, sims)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Try out my Skip-Gram Batcher!')
    parser.add_argument('--task_name', type=str, default='vanilla_zerou_72k',
                        help='Name of the folder with pretrained word vectors. Either '+\
                              '"vanilla_zerou_72k", or "negative_sampling_lr1e-3", '+\
                              '"negative_sampling_lr1e-4", "vanilla_zerou_100k"')
    args = parser.parse_args()

    save_dir = osp.join(config.model_dir, 'final', args.task_name)

    vectors_path = osp.join(save_dir, 'word_vectors.tsv')
    meta_path = osp.join(save_dir, 'meta.tsv')
    w2v = Word2Vec(vectors_path, meta_path)
    print(f'Word vectors were successfully loaded from {save_dir}')
    print('Solving analogy test: king - man + woman', end='\n\n')
    print(w2v.most_similar(['king', 'woman'], ['man']))