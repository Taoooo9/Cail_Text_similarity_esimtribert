import time
import numpy as np
import torch

from Units.units import find_maxlen, find_maxlennum, find_char_maxlen
from pytorch_transformers import BertTokenizer


def create_batch(tra_data, tra_word_vocab, batch_size, config, shuffle=True):
    print('create_batch is start')
    start_time = time.time()
    batch_word_data = []
    batch_char_data = []
    if shuffle:
        np.random.shuffle(tra_data)

    unit = []
    instances = []
    for instance in tra_data:
        instances.append(instance)
        if len(instances) == batch_size:
            unit.append(instances)
            instances = []

    if len(instances) > 0:
        unit.append(instances)

    for batch in unit:
        one_word_data = bert_word_data_variable(batch, config)
        batch_word_data.append(one_word_data)
        # one_char_data = pair_char_data_variable(batch, tra_char_vocab, config)
        # batch_char_data.append(one_char_data)

    print('the create_batch use:{:.2f} S'.format(time.time() - start_time))
    return batch_word_data


def bert_word_data_variable(batch, config):
    tokenizer = BertTokenizer.from_pretrained('Data/ms/.')
    batch_size = len(batch) * 2
    src_premise_matrix = np.zeros((batch_size, config.max_sen_len + 2))
    src_hypothesis_matrix = np.zeros((batch_size, config.max_sen_len + 2))
    p_mask = np.zeros((batch_size, config.max_sen_len + 2))
    h_mask = np.zeros((batch_size, config.max_sen_len + 2))
    tag_matrix = np.zeros(batch_size)
    for idx, instance in enumerate(batch):
        premise = tokenizer.encode(instance[0])
        hypothesis_b = tokenizer.encode(instance[1])
        hypothesis_c = tokenizer.encode(instance[2])
        while len(premise) > config.max_sen_len:
            premise = premise[len(premise) - config.max_sen_len:]
        while len(hypothesis_b) > config.max_sen_len:
            hypothesis_b = hypothesis_b[len(hypothesis_b) - config.max_sen_len:]
        while len(hypothesis_c) > config.max_sen_len:
            hypothesis_c = hypothesis_c[len(hypothesis_c) - config.max_sen_len:]
        premise.insert(0, 101)
        premise.append(102)
        p_len = len(premise)

        hypothesis_b.insert(0, 101)
        hypothesis_b.append(102)
        hb_len = len(hypothesis_b)

        hypothesis_c.insert(0, 101)
        hypothesis_c.append(102)
        hc_len = len(hypothesis_c)

        for jdx in range(p_len):
            src_premise_matrix[idx * 2][jdx] = premise[jdx]
            src_premise_matrix[idx * 2 + 1][jdx] = premise[jdx]
            p_mask[idx * 2][jdx] = 1
            p_mask[idx * 2 + 1][jdx] = 1
        for kdx in range(hb_len):
            src_hypothesis_matrix[idx * 2][kdx] = hypothesis_b[kdx]
            h_mask[idx * 2][kdx] = 1
        for gdx in range(hc_len):
            src_hypothesis_matrix[idx * 2 + 1][gdx] = hypothesis_c[gdx]
            h_mask[idx * 2 + 1][gdx] = 1
        tag_matrix[idx * 2] = 1
        tag_matrix[idx * 2 + 1] = 0
    src_premise_matrix = torch.from_numpy(src_premise_matrix).long()
    src_hypothesis_matrix = torch.from_numpy(src_hypothesis_matrix).long()
    p_mask = torch.from_numpy(p_mask).float()
    h_mask = torch.from_numpy(h_mask).float()
    tag_matrix = torch.from_numpy(tag_matrix).long()
    if config.use_cuda:
        src_premise_matrix = src_premise_matrix.cuda()
        src_hypothesis_matrix = src_hypothesis_matrix.cuda()
        p_mask = p_mask.cuda()
        h_mask = h_mask.cuda()
        tag_matrix = tag_matrix.cuda()
    return [src_premise_matrix, src_hypothesis_matrix, p_mask, h_mask, tag_matrix]


def pair_word_data_variable(batch, tra_word_vocab, config):
    count = 0
    tokenizer = BertTokenizer.from_pretrained('Data/ms/.')
    premise_length = []
    hypothesis_length = []
    batch_size = len(batch)
    src_premise_matrix = np.zeros((batch_size * 2, config.max_sen_len))
    src_hypothesis_matrix = np.zeros((batch_size * 2, config.max_sen_len))
    premise_matrix_mask = np.zeros((batch_size * 2, config.max_sen_len))
    hypothesis_matrix_mask = np.zeros((batch_size * 2, config.max_sen_len))
    tag_matrix = []
    for idx, instance in enumerate(batch):
        for kdx, sentences in enumerate(instance[0:3]):
            sentences = tokenizer.encode(sentences)
            if len(sentences) > 510:
                sentences = sentences[len(sentences) - 510:]
            if kdx == 0:
                premise_length.append(len(sentences) + 2)
                premise_length.append(len(sentences) + 2)
                src_premise_matrix[idx * 2][0] = 101
                src_premise_matrix[idx * 2 + 1][0] = 101
                premise_matrix_mask[idx * 2][0] = 1
                premise_matrix_mask[idx * 2 + 1][0] = 1
                for gdx, value in enumerate(sentences, 1):
                    src_premise_matrix[idx*2][gdx] = value
                    src_premise_matrix[idx*2+1][gdx] = value
                    premise_matrix_mask[idx*2][gdx] = 1
                    premise_matrix_mask[idx*2+1][gdx] = 1
                    count = gdx
                src_premise_matrix[idx * 2][count + 1] = 102
                src_premise_matrix[idx * 2 + 1][count + 1] = 102
                premise_matrix_mask[idx * 2][count + 1] = 1
                premise_matrix_mask[idx * 2 + 1][count + 1] = 1

            elif kdx == 1:
                tag_matrix.append(1)
                hypothesis_length.append(len(sentences) + 2)
                src_hypothesis_matrix[idx * 2][0] = 101
                hypothesis_matrix_mask[idx * 2][0] = 1
                for gdx, value in enumerate(sentences, 1):
                    src_hypothesis_matrix[idx * 2][gdx] = value
                    hypothesis_matrix_mask[idx * 2][gdx] = 1
                    count = gdx
                src_hypothesis_matrix[idx * 2][count + 1] = 102
                hypothesis_matrix_mask[idx * 2][count + 1] = 1

            else:
                tag_matrix.append(0)
                hypothesis_length.append(len(sentences) + 2)
                src_hypothesis_matrix[idx * 2 + 1][0] = 101
                hypothesis_matrix_mask[idx * 2 + 1][0] = 1
                for gdx, value in enumerate(sentences, 1):
                    src_hypothesis_matrix[idx * 2 + 1][gdx] = value
                    hypothesis_matrix_mask[idx * 2 + 1][gdx] = 1
                    count = gdx
                src_hypothesis_matrix[idx * 2 + 1][count + 1] = 102
                hypothesis_matrix_mask[idx * 2 + 1][count + 1] = 1

    src_premise_matrix = torch.from_numpy(src_premise_matrix).long()
    premise_matrix_mask = torch.from_numpy(premise_matrix_mask).float()
    src_hypothesis_matrix = torch.from_numpy(src_hypothesis_matrix).long()
    hypothesis_matrix_mask = torch.from_numpy(hypothesis_matrix_mask).float()
    tag_matrix = torch.tensor(tag_matrix).long()
    premise_length = torch.tensor(premise_length).long()
    hypothesis_length = torch.tensor(hypothesis_length).long()
    if config.use_cuda:
        src_premise_matrix = src_premise_matrix.cuda()
        premise_matrix_mask = premise_matrix_mask.cuda()
        src_hypothesis_matrix = src_hypothesis_matrix.cuda()
        hypothesis_matrix_mask = hypothesis_matrix_mask.cuda()
        tag_matrix = tag_matrix.cuda()
        premise_length = premise_length.cuda()
        hypothesis_length = hypothesis_length.cuda()
    return [src_premise_matrix, src_hypothesis_matrix, premise_length, hypothesis_length
            , premise_matrix_mask, hypothesis_matrix_mask, tag_matrix]


def pair_char_data_variable(batch, tra_char_vocab, config):
    batch_size = len(batch)
    max_premise_len, max_premise_char_len, max_hypothesis_len, max_hypothesis_char_len = find_char_maxlen(batch)
    src_premise_matrix = np.zeros((batch_size * 2, max_premise_len, max_premise_char_len))
    src_hypothesis_matrix = np.zeros((batch_size * 2, max_hypothesis_len, max_hypothesis_char_len))
    for idx, instance in enumerate(batch):
        for kdx, sentence in enumerate(instance[0:3]):
            if kdx == 0:
                for jdx, word in enumerate(sentence):
                    for gdx, char in enumerate(word):
                        src_premise_matrix[idx * 2][jdx][gdx] = tra_char_vocab.word2id(char)
                        src_premise_matrix[idx * 2 + 1][jdx][gdx] = tra_char_vocab.word2id(char)
            elif kdx == 1:
                for jdx, word in enumerate(sentence):
                    for gdx, char in enumerate(word):
                        src_hypothesis_matrix[idx * 2][jdx][gdx] = tra_char_vocab.word2id(char)
            else:
                for jdx, word in enumerate(sentence):
                    for gdx, char in enumerate(word):
                        src_hypothesis_matrix[idx * 2 + 1][jdx][gdx] = tra_char_vocab.word2id(char)
    src_premise_matrix = torch.from_numpy(src_premise_matrix).long()
    src_hypothesis_matrix = torch.from_numpy(src_hypothesis_matrix).long()
    if config.use_cuda:
        src_premise_matrix = src_premise_matrix.cuda()
        src_hypothesis_matrix = src_hypothesis_matrix.cuda()
    return [src_premise_matrix, src_hypothesis_matrix]


def create_sen_batch(tra_data, tra_fact_vocab, config, shuffle=True):
    print('create_batch is start')
    start_time = time.time()
    batch_data = []
    if shuffle:
        np.random.shuffle(tra_data)

    unit = []
    instances = []
    for instance in tra_data:
        instances.append(instance)
        if len(instances) == config.batch_size:
            unit.append(instances)
            instances = []

    if len(instances) > 0:
        unit.append(instances)

    for batch in unit:
        one_data = pair_sen_data_variable(batch, tra_fact_vocab, config)
        batch_data.append(one_data)

    print('the create_batch use:{:.2f} S'.format(time.time() - start_time))
    return batch_data


# def pair_sen_data_variable(batch, tra_fact_vocab, config):
#     batch_size = len(batch) * 2
#     sen_premise_length = []
#     sen_hypothesis_length = []
#     sen_per_premise_length = []
#     sen_per_hypothesis_length = []
#     max_data_length, max_sen_num = find_maxlennum(batch)
#     src_sen_premise_matrix = np.zeros((batch_size, max_sen_num, max_data_length))
#     src_sen_hypothesis_matrix = np.zeros((batch_size, max_sen_num, max_data_length))
#     for idx, instance in enumerate(batch):
#         for kdx, sentences in enumerate(instance[0:3]):
#
#             if kdx == 0:
#                 sen_premise_length.append(len(sentences))
#                 sen_premise_length.append(len(sentences))
#                 for wdx, sentence in enumerate(sentences):
#                     sen_per_premise_length.append(len(sentence))
#                     sen_per_premise_length.append(len(sentence))
#                     sentence = tra_fact_vocab.word2id(sentence)
#                     for jdx, value in enumerate(sentence):
#                         src_sen_premise_matrix[idx*2][wdx][jdx] = value
#                         src_sen_premise_matrix[idx*2+1][wdx][jdx] = value
#                     # if wdx + 1 == len(sentences):
#                     #     for gdx in range(wdx+1, max_sen_num):
#                     #         sen_per_premise_length.append(0)
#                     #         sen_per_premise_length.append(0)
#             elif kdx == 1:
#                 sen_hypothesis_length.append(len(sentences))
#                 for wdx, sentence in enumerate(sentences):
#                     sen_per_hypothesis_length.append(len(sentence))
#                     sentence = tra_fact_vocab.word2id(sentence)
#                     for jdx, value in enumerate(sentence):
#                         src_sen_hypothesis_matrix[idx * 2][wdx][jdx] = value
#                     # if wdx + 1 == len(sentences):
#                     #     for gdx in range(wdx + 1, max_sen_num):
#                     #         sen_per_hypothesis_length.append(0)
#             else:
#                 sen_hypothesis_length.append(len(sentences))
#                 for wdx, sentence in enumerate(sentences):
#                     sen_per_hypothesis_length.append(len(sentence))
#                     sentence = tra_fact_vocab.word2id(sentence)
#                     for jdx, value in enumerate(sentence):
#                         src_sen_hypothesis_matrix[idx * 2 + 1][wdx][jdx] = value
#                     # if wdx + 1 == len(sentences):
#                     #     for gdx in range(wdx + 1, max_sen_num):
#                     #         sen_per_hypothesis_length.append(0)
#
#     src_sen_premise_matrix = torch.from_numpy(src_sen_premise_matrix).long()
#     src_sen_hypothesis_matrix = torch.from_numpy(src_sen_hypothesis_matrix).long()
#     sen_premise_length = torch.tensor(sen_premise_length)
#     sen_hypothesis_length = torch.tensor(sen_hypothesis_length)
#     sen_per_premise_length = torch.tensor(sen_per_premise_length)
#     sen_per_hypothesis_length = torch.tensor(sen_per_hypothesis_length)
#     if config.use_cuda:
#         src_sen_premise_matrix = src_sen_premise_matrix.cuda()
#         src_sen_hypothesis_matrix = src_sen_hypothesis_matrix.cuda()
#         sen_premise_length = sen_premise_length.cuda()
#         sen_hypothesis_length = sen_hypothesis_length.cuda()
#         sen_per_premise_length = sen_per_premise_length.cuda()
#         sen_per_hypothesis_length = sen_per_hypothesis_length.cuda()
#     return [src_sen_premise_matrix, src_sen_hypothesis_matrix, sen_premise_length, sen_hypothesis_length, sen_per_premise_length,
#             sen_per_hypothesis_length]


def pair_sen_data_variable(batch, tra_fact_vocab, config):
    batch_size = len(batch) * 2
    sen_premise_length = []
    sen_hypothesis_length = []
    sen_per_premise_length = []
    sen_per_hypothesis_length = []
    max_data_length, max_sen_p_num, max_sen_h_num = find_maxlennum(batch)
    src_sen_premise_matrix = np.zeros((batch_size, max_sen_p_num, max_data_length))
    src_sen_premise_mask = np.zeros((batch_size, max_sen_p_num))
    src_sen_hypothesis_matrix = np.zeros((batch_size, max_sen_h_num, max_data_length))
    src_sen_hypothesis_mask = np.zeros((batch_size, max_sen_h_num))
    for idx, instance in enumerate(batch):
        for kdx, sentences in enumerate(instance[0:3]):
            if kdx == 0:
                sen_premise_length.append(len(sentences))
                sen_premise_length.append(len(sentences))
                for wdx, sentence in enumerate(sentences):
                    src_sen_premise_mask[idx*2][wdx] = 1
                    src_sen_premise_mask[idx*2 + 1][wdx] = 1
                    sen_per_premise_length.append(len(sentence))
                    sen_per_premise_length.append(len(sentence))
                    sentence = tra_fact_vocab.word2id(sentence)
                    for jdx, value in enumerate(sentence):
                        src_sen_premise_matrix[idx*2][wdx][jdx] = value
                        src_sen_premise_matrix[idx*2+1][wdx][jdx] = value
                    if wdx + 1 == len(sentences):
                        for gdx in range(wdx+1, max_sen_p_num):
                            src_sen_premise_matrix[idx * 2][wdx][gdx] = 3
                            src_sen_premise_matrix[idx * 2 + 1][wdx][gdx] = 3
                            sen_per_premise_length.append(1)
                            sen_per_premise_length.append(1)
            elif kdx == 1:
                sen_hypothesis_length.append(len(sentences))
                for wdx, sentence in enumerate(sentences):
                    src_sen_hypothesis_mask[idx*2][wdx] = 1
                    sen_per_hypothesis_length.append(len(sentence))
                    sentence = tra_fact_vocab.word2id(sentence)
                    for jdx, value in enumerate(sentence):
                        src_sen_hypothesis_matrix[idx * 2][wdx][jdx] = value
                    if wdx + 1 == len(sentences):
                        for gdx in range(wdx + 1, max_sen_h_num):
                            src_sen_hypothesis_matrix[idx * 2][wdx][gdx] = 3
                            sen_per_hypothesis_length.append(1)
            else:
                sen_hypothesis_length.append(len(sentences))
                for wdx, sentence in enumerate(sentences):
                    src_sen_hypothesis_mask[idx * 2 + 1][wdx] = 1
                    sen_per_hypothesis_length.append(len(sentence))
                    sentence = tra_fact_vocab.word2id(sentence)
                    for jdx, value in enumerate(sentence):
                        src_sen_hypothesis_matrix[idx * 2 + 1][wdx][jdx] = value
                    if wdx + 1 == len(sentences):
                        for gdx in range(wdx + 1, max_sen_h_num):
                            src_sen_hypothesis_matrix[idx * 2 + 1][wdx][gdx] = 3
                            sen_per_hypothesis_length.append(1)

    src_sen_premise_matrix = torch.from_numpy(src_sen_premise_matrix).long()
    src_sen_hypothesis_matrix = torch.from_numpy(src_sen_hypothesis_matrix).long()
    src_sen_premise_mask = torch.from_numpy(src_sen_premise_mask).float()
    src_sen_hypothesis_mask = torch.from_numpy(src_sen_hypothesis_mask).float()
    sen_premise_length = torch.tensor(sen_premise_length)
    sen_hypothesis_length = torch.tensor(sen_hypothesis_length)
    sen_per_premise_length = torch.tensor(sen_per_premise_length)
    sen_per_hypothesis_length = torch.tensor(sen_per_hypothesis_length)
    if config.use_cuda:
        src_sen_premise_matrix = src_sen_premise_matrix.cuda()
        src_sen_hypothesis_matrix = src_sen_hypothesis_matrix.cuda()
        src_sen_premise_mask = src_sen_premise_mask.cuda()
        src_sen_hypothesis_mask = src_sen_hypothesis_mask.cuda()
        sen_premise_length = sen_premise_length.cuda()
        sen_hypothesis_length = sen_hypothesis_length.cuda()
        sen_per_premise_length = sen_per_premise_length.cuda()
        sen_per_hypothesis_length = sen_per_hypothesis_length.cuda()
    return [src_sen_premise_matrix, src_sen_hypothesis_matrix, sen_premise_length, sen_hypothesis_length, sen_per_premise_length,
            sen_per_hypothesis_length, src_sen_premise_mask, src_sen_hypothesis_mask]
