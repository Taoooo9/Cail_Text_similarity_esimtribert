from configparser import ConfigParser


class Config(object):

    def __init__(self, config_file):
        config = ConfigParser()
        config.read(config_file)
        for section in config.sections():
            for k, v in config.items(section):
                print(k, ":", v)
        self._config = config
        self.config_file = config_file
        config.write(open(config_file, 'w+'))

    def add_args(self, section, key, value):
        if self._config.has_section(section):
            print('This is a section already.')
        else:
            print('Now, we will add a new section.')
            self._config.add_section(section)
        if self._config.has_option(section, key):
            self._config.set(section, key, value)
            print('Add parameter successfully.')
        self._config.write(open(self.config_file, 'w'))

    # Data
    @property
    def train_file(self):
        return self._config.get('Data', 'train_file')

    @property
    def dev_file(self):
        return self._config.get('Data', 'dev_file')

    @property
    def embedding_file(self):
        return self._config.get('Data', 'embedding_file')

    @property
    def stop_words(self):
        return self._config.get('Data', 'stop_words')

    # Save
    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @property
    def save_pkl_path(self):
        return self._config.get('Save', 'save_pkl_path')

    @property
    def save_analysis_pkl_path(self):
        return self._config.get('Save', 'save_analysis_pkl_path')

    @property
    def save_model_path(self):
        return self._config.get('Save', 'save_model_path')

    @property
    def save_vocab_path(self):
        return self._config.get('Save', 'save_vocab_path')

    @property
    def save_vocab_path_fact(self):
        return self._config.get('Save', 'save_vocab_path_fact')

    @property
    def save_vocab_path_accusation(self):
        return self._config.get('Save', 'save_vocab_path_accusation')

    @property
    def bert_model_pkl(self):
        return self._config.get('Save', 'bert_model_pkl')

    @property
    def esim_model_pkl(self):
        return self._config.get('Save', 'esim_model_pkl')

    @property
    def char_model_pkl(self):
        return self._config.get('Save', 'char_model_pkl')

    @property
    def train_data_word_pkl(self):
        return self._config.get('Save', 'train_data_word_pkl')

    @property
    def train_data_analysis_pkl(self):
        return self._config.get('Save', 'train_data_analysis_pkl')

    @property
    def dev_data_word_pkl(self):
        return self._config.get('Save', 'dev_data_word_pkl')

    @property
    def test_data_word_pkl(self):
        return self._config.get('Save', 'test_data_word_pkl')

    @property
    def train_data_sen_pkl(self):
        return self._config.get('Save', 'train_data_sen_pkl')

    @property
    def embedding_pkl(self):
        return self._config.get('Save', 'embedding_pkl')

    @property
    def dev_data_sen_pkl(self):
        return self._config.get('Save', 'dev_data_sen_pkl')

    @property
    def test_data_sen_pkl(self):
        return self._config.get('Save', 'test_data_sen_pkl')

    @property
    def fact_sen_vocab(self):
        return self._config.get('Save', 'fact_sen_vocab')

    @property
    def fact_word_vocab(self):
        return self._config.get('Save', 'fact_word_vocab')

    @property
    def fact_char_vocab(self):
        return self._config.get('Save', 'fact_char_vocab')

    # Data_analysis
    @property
    def analysis_data(self):
        return self._config.getboolean('Data_analysis', 'analysis_data')

    # Train
    @property
    def use_cuda(self):
        return self._config.getboolean('Train', 'use_cuda')

    @property
    def epoch(self):
        return self._config.getint('Train', 'epoch')

    @property
    def batch_size(self):
        return self._config.getint('Train', 'batch_size')

    @property
    def dev_batch_size(self):
        return self._config.getint('Train', 'dev_batch_size')

    @property
    def use_lr_decay(self):
        return self._config.getboolean('Train', 'use_lr_decay')

    @property
    def clip_max_norm_use(self):
        return self._config.getboolean('Train', 'clip_max_norm_use')

    @property
    def test_interval(self):
        return self._config.getint('Train', 'test_interval')

    @property
    def early_stop(self):
        return self._config.getint('Train', 'early_stop')

    @property
    def update_every(self):
        return self._config.getint('Train', 'update_every')

    @property
    def n_fold(self):
        return self._config.getint('Train', 'n_fold')

    @property
    def less_triplet_loss(self):
        return self._config.getboolean('Train', 'less_triplet_loss')

    # Data_loader
    @property
    def data_cut(self):
        return self._config.getboolean('Data_loader', 'data_cut')

    @property
    def data_cut_k(self):
        return self._config.getint('Data_loader', 'data_cut_k')

    @property
    def stop_word(self):
        return self._config.getboolean('Data_loader', 'stop_word')

    @property
    def read_sen(self):
        return self._config.getboolean('Data_loader', 'read_sen')

    @property
    def read_char(self):
        return self._config.getboolean('Data_loader', 'read_char')

    # Model
    @property
    def embedding_char_dim(self):
        return self._config.getint('Model', 'embedding_char_dim')

    @property
    def embedding_word_dim(self):
        return self._config.getint('Model', 'embedding_word_dim')

    @property
    def embedding_sen_dim(self):
        return self._config.getint('Model', 'embedding_sen_dim')

    @property
    def embedding_word_num(self):
        return self._config.getint('Model', 'embedding_word_num')

    @property
    def embedding_sen_num(self):
        return self._config.getint('Model', 'embedding_sen_num')

    @property
    def embedding_char_num(self):
        return self._config.getint('Model', 'embedding_char_num')

    @property
    def input_channels(self):
        return self._config.getint('Model', 'input_channels')

    @property
    def hidden_size(self):
        return self._config.getint('Model', 'hidden_size')

    @property
    def kernel_size(self):
        return self._config.get('Model', 'kernel_size')

    @property
    def kernel_num(self):
        return self._config.getint('Model', 'kernel_num')

    @property
    def dropout(self):
        return self._config.getfloat('Model', 'dropout')

    @property
    def class_num(self):
        return self._config.getint('Model', 'class_num')

    @property
    def which_model(self):
        return self._config.get('Model', 'which_model')

    @property
    def learning_algorithm(self):
        return self._config.get('Model', 'learning_algorithm')

    @property
    def bert_lr(self):
        return self._config.getfloat('Model', 'bert_lr')

    @property
    def esim_lr(self):
        return self._config.getfloat('Model', 'esim_lr')

    @property
    def weight_decay(self):
        return self._config.getfloat('Model', 'weight_decay')

    @property
    def accusation_num(self):
        return self._config.getint('Model', 'accusation_num')

    @property
    def lr_rate_decay(self):
        return self._config.getfloat('Model', 'lr_rate_decay')

    @property
    def margin(self):
        return self._config.getfloat('Model', 'margin')

    @property
    def p(self):
        return self._config.getint('Model', 'p')

    @property
    def patience(self):
        return self._config.getint('Model', 'patience')

    @property
    def epsilon(self):
        return self._config.getfloat('Model', 'epsilon')

    @property
    def factor(self):
        return self._config.getfloat('Model', 'factor')

    @property
    def min_lr(self):
        return self._config.getfloat('Model', 'min_lr')

    @property
    def pre_embedding(self):
        return self._config.getboolean('Model', 'pre_embedding')

    @property
    def correct_bias(self):
        return self._config.getboolean('Model', 'correct_bias')

    @property
    def which(self):
        return self._config.get('Model', 'which')

    @property
    def hidden_style(self):
        return self._config.get('Model', 'hidden_style')

    @property
    def max_sen_len(self):
        return self._config.getint('Model', 'max_sen_len')

    @property
    def k(self):
        return self._config.getint('Model', 'k')

