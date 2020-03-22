from collections import Counter, defaultdict, Iterable
from itertools import chain
from logging import getLogger
from typing import Optional, Tuple, List

import numpy as np

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import zero_pad, is_str_batch, flatten_str_batch
from deeppavlov.core.models.estimator import Estimator

log = getLogger(__name__)


@register('simple_vocab')
class SimpleVocabulary(Estimator):
    """Implements simple vocabulary.
    Parameters:
        special_tokens: tuple of tokens that shouldn't be counted.
        max_tokens: upper bound for number of tokens in the vocabulary.
        min_freq: minimal count of a token (except special tokens).
        pad_with_zeros: if True, then batch of elements will be padded with zeros up to length of
            the longest element in batch.
        unk_token: label assigned to unknown tokens.
        freq_drop_load: if True, then frequencies of tokens are set to min_freq on the model load.
        """

    def __init__(self,
                 special_tokens: Tuple[str, ...] = tuple(),
                 max_tokens: int = 2 ** 30,
                 min_freq: int = 0,
                 pad_with_zeros: bool = False,
                 unk_token: Optional[str] = None,
                 freq_drop_load: Optional[bool] = None,
                 *args,
                 **kwargs):
        super().__init__(**kwargs)
        self.special_tokens = special_tokens
        self._max_tokens = max_tokens
        self._min_freq = min_freq
        self._pad_with_zeros = pad_with_zeros
        self.unk_token = unk_token
        self.freq_drop_load = freq_drop_load
        self.reset()
        if self.load_path:
            self.load()

    def fit(self, *args):
        self.reset()
        tokens = chain(*args)
        # filter(None, <>) -- to filter empty tokens
        self.freqs = Counter(filter(None, flatten_str_batch(tokens)))
        for special_token in self.special_tokens:
            self._t2i[special_token] = self.count
            self._i2t.append(special_token)
            self.count += 1
        for token, freq in self.freqs.most_common()[:self._max_tokens]:
            if token in self.special_tokens:
                continue
            if freq >= self._min_freq:
                self._t2i[token] = self.count
                self._i2t.append(token)
                self.count += 1

    def _add_tokens_with_freqs(self, tokens, freqs):
        self.freqs = Counter()
        self.freqs.update(dict(zip(tokens, freqs)))
        for token, freq in zip(tokens, freqs):
            if freq >= self._min_freq or token in self.special_tokens:
                self._t2i[token] = self.count
                self._i2t.append(token)
                self.count += 1

    def __call__(self, batch, is_top=True, **kwargs):
        if isinstance(batch, Iterable) and not isinstance(batch, str):
            looked_up_batch = [self(sample, is_top=False) for sample in batch]
        else:
            return self[batch]
        if self._pad_with_zeros and is_top and not is_str_batch(looked_up_batch):
            looked_up_batch = zero_pad(looked_up_batch)

        return looked_up_batch

    def save(self):
        log.info("[saving vocabulary to {}]".format(self.save_path))
        with self.save_path.open('wt', encoding='utf8') as f:
            for n in range(len(self)):
                token = self._i2t[n]
                cnt = self.freqs[token]
                f.write('{}\t{:d}\n'.format(token, cnt))

    def serialize(self) -> List[Tuple[str, int]]:
        return [(token, self.freqs[token]) for token in self._i2t]

    def load(self):
        self.reset()
        if self.load_path:
            if self.load_path.is_file():
                log.info("[loading vocabulary from {}]".format(self.load_path))
                tokens, counts = [], []
                for ln in self.load_path.open('r', encoding='utf8'):
                    token, cnt = self.load_line(ln)
                    tokens.append(token)
                    counts.append(int(cnt))
                self._add_tokens_with_freqs(tokens, counts)
            elif not self.load_path.parent.is_dir():
                raise ConfigError("Provided `load_path` for {} doesn't exist!".format(
                    self.__class__.__name__))
        else:
            raise ConfigError("`load_path` for {} is not provided!".format(self))

    def deserialize(self, data: List[Tuple[str, int]]) -> None:
        self.reset()
        if data:
            tokens, counts = zip(*data)
            self._add_tokens_with_freqs(tokens, counts)

    def load_line(self, ln):
        if self.freq_drop_load:
            token = ln.strip().split()[0]
            cnt = self._min_freq
        else:
            token, cnt = ln.rsplit('\t', 1)
        return token, cnt

    @property
    def len(self):
        return len(self)

    def keys(self):
        return (self[n] for n in range(self.len))

    def values(self):
        return list(range(self.len))

    def items(self):
        return zip(self.keys(), self.values())

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._i2t[key]
        elif isinstance(key, str):
            return self._t2i[key]
        else:
            raise NotImplementedError("not implemented for type `{}`".format(type(key)))

    def __contains__(self, item):
        return item in self._t2i

    def __len__(self):
        return len(self._i2t)

    def reset(self):
        self.freqs = None
        unk_index = 0
        if self.unk_token in self.special_tokens:
            unk_index = self.special_tokens.index(self.unk_token)
        self._t2i = defaultdict(lambda: unk_index)
        self._i2t = []
        self.count = 0

    def idxs2toks(self, idxs):
        return [self[idx] for idx in idxs]\
# ============================================================================



# версия из учебника
class Vocabulary(object):
    """ Класс, предназначенный для обработки текста и извлечения Vocabulary
    для отображения
    """
    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        Аргументы:
        token_to_idx (dict): готовый ассоциативный массив соответствий
        токенов индексам add_unk (bool): флаг, указывающий,
        нужно ли добавлять токен UNK
        unk_token (str): добавляемый в словарь токен UNK
        """
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token
        for token, idx in self._token_to_idx.items()}
        self._add_unk = add_unk
        self._unk_token = unk_token
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def to_serializable(self):
        """ Возвращает словарь с возможностью сериализации """
        return {'token_to_idx': self._token_to_idx,
            'add_unk': self._add_unk,
            'unk_token': self._unk_token}

    @classmethod
    def from_serializable(cls, contents):
        """ Создает экземпляр Vocabulary на основе сериализованного словаря """
        return cls(**contents)

    def add_token(self, token):
        """ Обновляет словари отображения, добавляя в них токен.
        Аргументы:
        token (str): добавляемый в Vocabulary элемент
        Возвращает:
        index (int): соответствующее токену целочисленное значение
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def lookup_token(self, token):
        """ Извлекает соответствующий токену индекс
        или индекс UNK, если токен не найден.
        Аргументы:
        token (str): токен для поиска
        Возвращает:
        index (int): соответствующий токену индекс
        Примечания:
        'unk_index' должен быть >=0 (добавлено в Vocabulary)
        для должного функционирования UNK
        """
        if self.add_unk:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """ Возвращает соответствующий индексу токен
        Аргументы:
        index (int): индекс для поиска
        Возвращает:
        token (str): соответствующий индексу токен
        Генерирует:
        KeyError: если индекс не найден в Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)

