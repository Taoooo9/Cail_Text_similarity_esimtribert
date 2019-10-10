import os

from Units.units import *
from pytorch_transformers import BertTokenizer


class ReadData(object):

    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained('Data/ms/.')
        if not os.path.isdir(config.save_dir):
            os.makedirs(config.save_dir)
        if not os.path.isdir(config.save_pkl_path):
            os.makedirs(config.save_pkl_path)
        if config.stop_word:
            self.stop_words = stop_word(config.stop_words)

    def read_data(self):
        if os.path.isfile(self.config.train_data_word_pkl):
            tra_data = read_pkl(self.config.train_data_word_pkl)
        else:
            tra_data = self.load_data(self.config.train_file)
            pickle.dump(tra_data, open(self.config.train_data_word_pkl, 'wb'))
        if os.path.isfile(self.config.dev_data_word_pkl):
            dev_data = read_pkl(self.config.dev_data_word_pkl)
        else:
            dev_data = self.load_data(self.config.dev_file)
            pickle.dump(dev_data, open(self.config.dev_data_word_pkl, 'wb'))
        return tra_data, dev_data

    def load_data(self, file):
        data = []
        unit = []
        max_len = 0
        with open(file, encoding='utf-8') as f:
            for line in f.readlines():
                if line != '\n':
                    line = line.split(' ')
                    line = ''.join(line)
                    unit.append(line)
                    if max_len < len(line):
                        max_len = len(line)
                else:
                    unit.append(max_len)
                    data.append(unit)
                    unit = []
                    max_len = 0
            return data


class ReadSenData(object):

    def __init__(self, config):
        self.config = config
        if not os.path.isdir(config.save_dir):
            os.makedirs(config.save_dir)
        if not os.path.isdir(config.save_pkl_path):
            os.makedirs(config.save_pkl_path)
        if config.stop_word:
            self.stop_words = stop_word(config.stop_words)

    def read_data(self):
        if os.path.isfile(self.config.train_data_sen_pkl):
            tra_data = read_pkl(self.config.train_data_sen_pkl)
        else:
            tra_data = self.load_sen_data(self.config.train_file)
            pickle.dump(tra_data, open(self.config.train_data_sen_pkl, 'wb'))
        if os.path.isfile(self.config.dev_data_sen_pkl):
            dev_data = read_pkl(self.config.dev_data_sen_pkl)
        else:
            dev_data = self.load_sen_data(self.config.dev_file)
            pickle.dump(dev_data, open(self.config.dev_data_sen_pkl, 'wb'))
        return tra_data, dev_data

    def load_sen_data(self, file):
        data = []
        unit = []
        units = []
        max_len = 0
        max_num = 0
        count = 0
        sen_temp = []
        with open(file, encoding='utf-8') as f:
            for line in f.readlines():
                if line != '\n':
                    line = line.strip()
                    line = line.strip('。')
                    line = line.split('。')
                    for sentences in line:
                        sentences = sentences.strip('；')
                        sentences = sentences.split('；')
                        for sentence in sentences:
                            sentence = sentence.strip(' ')
                            sentence = sentence.split(' ')
                            if len(sen_temp) < 16:
                                if len(sen_temp) + len(sentence) > 23 and len(sen_temp) != 0:
                                    if len(sentence) > max_len:
                                        max_len = len(sentence)
                                    if len(sen_temp) > max_len:
                                        max_len = len(sen_temp)
                                    unit.append(sen_temp)
                                    unit.append(sentence)
                                    count += 2
                                    sen_temp = []
                                else:
                                    sen_temp.extend(sentence)
                                    if len(sen_temp) > max_len:
                                        max_len = len(sen_temp)
                                    if len(sen_temp) > 15:
                                        unit.append(sen_temp)
                                        count += 1
                                        sen_temp = []
                            else:
                                if len(sen_temp) > max_len:
                                    max_len = len(sen_temp)
                                unit.append(sen_temp)
                                count += 1
                                sen_temp = []
                    if len(sen_temp) > 0:
                        if len(sen_temp) > max_len:
                            max_len = len(sen_temp)
                        unit.append(sen_temp)
                        sen_temp = []
                        count += 1
                    units.append(unit)
                    if count > max_num:
                        max_num = count
                    unit = []
                    count = 0
                else:
                    units.append(max_len)
                    units.append(max_num)
                    data.append(units)
                    units = []
                    max_len = 0
                    max_num = 0
            return data




