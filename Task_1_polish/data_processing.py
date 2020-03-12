import numpy as np
from collections import Counter
from string import punctuation
import spacy
from nltk.tokenize import TweetTokenizer
from sortedcontainers import SortedSet


class Preprocess():
    def __init__(self, path, limit=5, window_size=5, mode='sg'):
        """
        Batcher for Skip-Gram or CBOW

        :param text: String without newline symbols
        :param limit: Don't put words with less amount into the dictonary
        :param window_size: Window size ^)
        :param mode: cbow or sg
        """
        with open(path) as f:
            self.text = f.read()

        self.limit = limit
        self.window_size = window_size
        self.mode = mode
        self.tokens = []
        self.vocabulary = set()
        self.word2index = dict()
        self.index2word = dict()
        self.X = []
        self.y = []

    @classmethod
    def from_file(cls, path, limit=5, window_size=5, mode='sg'):
        """
        Init Batcher from file
        :param path: Path to text file
        :param limit: Don't put words with less amount into the dictonary
        :param window_size: Window size ^)
        :param mode: cbow or sg
        :return: Batcher object
        """
        with open(path) as f:
            text = f.read()
        return cls(text, limit, window_size, mode)

    def _punctuation(self):
        return ''.join([c for c in self.text if c not in punctuation + '«»'])

    def _tokenize_for_Task_1(self):
        self.tokens = self.text.split()

    def _tokenize(self, lemmatization, lower_case):
        self.text = self.text[:93621304]
        nlp = spacy.load('en_core_web_lg')
        nlp.max_length = 93621305
        if lower_case:
            doc = nlp(self.text.lower())
        else:
            doc = nlp(self.text)

        if lemmatization:
            self.tokens = [token.lemma_ for token in doc]
        else:
            self.tokens = [str(token) for token in doc]

    def _tweet_tokenize(self):
        self.tokens = TweetTokenizer().tokenize(self.text.lower())

    def _clean(self, punctuation):
        if punctuation==False:
            self.text = self._punctuation()

    def _build_vocabulary(self):
        counter_words = Counter(self.tokens)
        self.vocabulary = {word for word, counts in counter_words.items() if counts >= self.limit}

    def _numericalize(self):
        counter_words = Counter(self.tokens)
        self.vocabulary = SortedSet({word for word, counts in counter_words.items() if counts >= self.limit})
        self.word2index = {w: i + 1 for i, w in enumerate(self.vocabulary)}
        self.word2index['UNK'] = 0
        self.index2word = {i + 1: w for i, w in enumerate(self.vocabulary)}
        self.index2word[0] = 'UNK'

    def _prepare_data(self):
        indexes = [self.word2index[x] if x in self.word2index.keys() else 0 for x in self.tokens]
        for i, index in enumerate(indexes[self.window_size:-self.window_size]):
            idx = i + self.window_size
            self.y.append(index)
            self.X.append(indexes[idx - self.window_size:idx] + indexes[idx + 1:idx + self.window_size + 1])
        self.y = np.array(self.y)
        self.X = np.vstack(self.X)
        if self.mode=='sg':
            temp =self.y
            self.y = self.X
            self.X = temp

    def _preprocess(self, punctuation=True, lemmatization=True, lower_case=True):
        self._clean(punctuation)
        self._tokenize_for_Task_1()
        self._build_vocabulary()
        self._numericalize()

    def _get_data(self):
        self._preprocess()
        self._prepare_data()
        return self.X, self.y, self.word2index, self.index2word

