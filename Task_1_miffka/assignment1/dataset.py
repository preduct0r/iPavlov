import argparse
from collections import Counter
import numpy as np
import os.path as osp
from pprint import pprint

import torch
from torch.utils.data import Dataset, DataLoader

from Task_1_miffka.assignment1.config import config


class SkipGramDataset(Dataset):
    def __init__(self, text_file, dict_size=100000, min_count=0, window_size=5, 
                 neg_sample_n=10, test_mode=False, test_size=1000, transform=None):
        '''
        Class Dataset for Word2Vec learning. Implements Skip-gram design and allows to 
            implement Negative Sampling in a separate batch generator class.
        '''
        with open(text_file) as fin:
            text = fin.read().strip()
        token_seq = list(text.split())
        if test_mode:
            token_seq = token_seq[:test_size]
            dict_size = test_size // 10

        counter = Counter(token_seq)
        most_common = counter.most_common(dict_size - 2)

        ### Calculate weights for negative sampling ###
        most_common = [(token, count) for token, count in most_common if count >= min_count]
        n_unk = sum([count for _, count in most_common])
        n_empty = window_size * 2
        most_common = most_common + [('UNK', n_unk), ('EMPTY', n_empty)]
        weights = np.asarray([count / (len(token_seq) + n_empty) for _, count in most_common])
        weights = np.power(weights, 0.75)
        self.weights = weights / np.sum(weights)
        
        most_common = [token for token, _ in most_common]
        self.dict_size = len(most_common)
        self.n_token_array = np.arange(self.dict_size)
        
        self.token2int = {token: idx for idx, token in enumerate(most_common)}
        self.int2token = {v: k for k, v in self.token2int.items()}
        self.token_seq = np.asarray(self.words_to_index(token_seq),
                                    dtype=np.int32)
        self.window_size = window_size
        self.neg_sample_n = neg_sample_n
        self.transform = transform
        
        
    def words_to_index(self, tokens):
        if type(tokens) not in {list, np.ndarray}:
            tokens = [tokens]
        return [self.token2int.get(token, self.token2int['UNK']) for token in tokens]
    
    def index_to_words(self, indices):
        if type(indices) not in {list, np.ndarray}:
            indices = [indices]
        return [self.int2token.get(idx, 'UNK') for idx in indices]
    
    def __len__(self):
        return self.token_seq.shape[0]
    
    def __getitem__(self, idx):
        x = [self.token_seq[idx]]
        y = np.concatenate((
            [self.token2int['EMPTY']] * np.clip(self.window_size - idx, 0, self.window_size),
            self.token_seq[np.clip(idx - self.window_size, 0, None):idx], 
            self.token_seq[idx + 1:idx + self.window_size + 1],
            [self.token2int['EMPTY']] * np.clip(idx + self.window_size - (len(self.token_seq) - 1), 0, self.window_size)
            ), axis=0).astype(np.int32)
        
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        
        return x, y


def collate_fn(batch):
    x, targets = zip(*batch)
    x = torch.LongTensor(x)
    targets = torch.LongTensor(targets)
    return x, targets
    

class Batcher(DataLoader):
    '''
    Class DataLoader for Word2Vec learning. Extends simple DataLoader with the function
        that allows to translate indices to words.
    '''
    def indices_to_words(self, indices):
        if type(indices) == torch.Tensor:
            indices = indices.cpu().numpy().astype(np.int32)
        elif type(indices) not in {list, np.ndarray}:
            indices = [indices]
        return [self.dataset.index_to_words(phrase) for phrase in indices]


class BatcherNS(object):
    def __init__(self, dataset, batch_size, collate_fn, shuffle=True):
        '''
        Class Batcher with Negative Sampling that allows to generate negative samples
            in a fast and simple manner.
        '''
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_token_array = dataset.n_token_array
        self.neg_sample_n = dataset.neg_sample_n
        self.word_weights = dataset.weights
        self.n_full_batches = len(self.dataset) // batch_size
        self.last_batch_size = len(self.dataset) % batch_size
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self._prepare_batches()
        

    def _prepare_batches(self):
        samples_ids = np.arange(len(self.dataset))
        if self.shuffle:
            samples_ids = np.random.permutation(samples_ids)
        batches_ids = list(samples_ids[:-self.last_batch_size].reshape(self.n_full_batches, 
                                                                       self.batch_size))
        last_batch_ids = samples_ids[-self.last_batch_size:]
        self.batches_ids = batches_ids + [last_batch_ids]

    def indices_to_words(self, indices):
        if type(indices) == torch.Tensor:
            indices = indices.cpu().numpy().astype(np.int32)
        elif type(indices) not in {list, np.ndarray}:
            indices = [indices]
        return [self.dataset.index_to_words(phrase) for phrase in indices]

    def __iter__(self):
        return self.batch_generator()

    def __len__(self):
        return len(self.batches_ids)

    def batch_generator(self):
        for i, batch_ids in enumerate(self.batches_ids):
            x, y = self.collate_fn([self.dataset[idx] for idx in batch_ids])
            neg = torch.LongTensor(np.random.choice(self.n_token_array, 
                                                    size=(batch_ids.shape[0], self.neg_sample_n), 
                                                    replace=True, 
                                                    p=self.word_weights))
            if i == len(self) - 1:
                self._prepare_batches()
            yield x, y, neg

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Try out my Skip-Gram Batcher!')
    parser.add_argument('--text_file', type=str, default=osp.join(config.data_dir, 'text8'),
                        help='Path to text file')
    parser.add_argument('--negative_sampling', action='store_true',
                        help='Type of the dataset, either SkipGramDataset or SkipGramNSDataset')
    parser.add_argument('--dict_size', type=int, default=100000,
                        help='Size of the dictionary - how many unique words to fetch')
    parser.add_argument('--min_count', type=int, default=5,
                        help='Minimal count of the single word to be included in dictionary')
    parser.add_argument('--window_size', type=int, default=5,
                        help='Size of the window for context retrieval')
    parser.add_argument('--neg_sample_n', type=int, default=10,
                        help='Number of the negative samples for each word')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size')
    args = parser.parse_args()
    
    dataset = SkipGramDataset(args.text_file, dict_size=args.dict_size, min_count=args.min_count,
                              window_size=args.window_size, neg_sample_n=args.neg_sample_n)
    print('Size of the dictionary', dataset.dict_size)

    if not args.negative_sampling:
        loader = Batcher(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    else:
        loader = BatcherNS(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    for i, (batch) in enumerate(loader):
        if i > 0:
            break

        pprint(loader.indices_to_words(batch[0]))
        pprint(loader.indices_to_words(batch[1]))
        if args.negative_sampling:
            pprint(loader.indices_to_words(batch[2]))
