import os
from collections import Counter
from Units.units import *
UNK, PAD = 0, 1
PAD_S, UNK_S = '<pad>', '<unk>'


class FactWordVocab(object):

    def __init__(self, all_data, config):
        self.w2i = {}
        self.UNK = 1
        word_counter = Counter()
        for data in all_data:
            for line in data:
                for sentences in line[0:3]:
                    for word in sentences:
                        word_counter[word] += 1
        most_word = [k for k, v in word_counter.most_common()]
        self.i2w = [PAD_S, UNK_S] + most_word
        for idx, word in enumerate(self.i2w):
            self.w2i[word] = idx

        config.add_args('Model', 'embedding_word_num', str(self.getsize))

    def word2id(self, xx):
        if isinstance(xx, list):
            return [self.w2i.get(word, UNK) for word in xx]
        else:
            return self.w2i.get(xx)

    def id2word(self, xx):
        if isinstance(xx, list):
            return [self.i2w[idx] for idx in xx]
        else:
            return self.i2w[xx]

    @property
    def getsize(self):
        return len(self.i2w)


class FactSenVocab(object):

    def __init__(self, all_data, config):
        self.w2i = {}
        word_counter = Counter()
        for data in all_data:
            for line in data:
                for sentences in line[0:3]:
                    for sentence in sentences:
                        for word in sentence:
                            word_counter[word] += 1
        most_word = [k for k, v in word_counter.most_common()]
        self.i2w = [PAD_S, UNK_S] + most_word
        for idx, word in enumerate(self.i2w):
            self.w2i[word] = idx

        config.add_args('Model', 'embedding_sen_num', str(self.getsize))

    def word2id(self, xx):
        if isinstance(xx, list):
            return [self.w2i.get(word, UNK) for word in xx]
        else:
            return self.w2i.get(xx)

    def id2word(self, xx):
        if isinstance(xx, list):
            return [self.i2w[idx] for idx in xx]
        else:
            return self.i2w[xx]

    @property
    def getsize(self):
        return len(self.i2w)


class FactCharVocab(object):

    def __init__(self, all_data, config):
        self.w2i = {}
        word_counter = Counter()
        for data in all_data:
            for line in data:
                for sentences in line[0:3]:
                    for words in sentences[0]:
                        for char in words:
                            word_counter[char] += 1
        most_word = [k for k, v in word_counter.most_common()]
        self.i2w = [PAD_S, UNK_S] + most_word
        for idx, word in enumerate(self.i2w):
            self.w2i[word] = idx

        config.add_args('Model', 'embedding_char_num', str(self.getsize))

    def word2id(self, xx):
        if isinstance(xx, list):
            return [self.w2i.get(word, UNK) for word in xx]
        else:
            return self.w2i.get(xx)

    def id2word(self, xx):
        if isinstance(xx, list):
            return [self.i2w[idx] for idx in xx]
        else:
            return self.i2w[xx]

    @property
    def getsize(self):
        return len(self.i2w)


