import argparse

from Config.config import Config
from DataProcessing.data_read import *
from Model.BertModel import MyBertModel
from Vocab.vocab import *
from Train.train import train


if __name__ == '__main__':

    # seed
    random_seed(520)

    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))

    # gpu
    gpu = torch.cuda.is_available()
    if gpu:
        print('The train will be using GPU.')
    else:
        print('The train will be using CPU.')
    print('CuDNN', torch.backends.cudnn.enabled)

    # config
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config_file', type=str, default='./Config/config.ini')
    args = arg_parser.parse_args()
    config = Config(args.config_file)
    if gpu:
        config.add_args('Train', 'use_cuda', 'True')

    # data_loader
    sem_data_loader = ReadSenData(config)
    tra_sen_data, dev_sen_data = sem_data_loader.read_data()
    word_data_loader = ReadData(config)
    tra_data, dev_data = word_data_loader.read_data()

    # vocab
    if os.path.isfile(config.fact_word_vocab):
        tra_fact_word_vocab = read_pkl(config.fact_word_vocab)
        tra_fact_char_vocab = read_pkl(config.fact_char_vocab)
        if config.read_sen:
            tra_fact_sen_vocab = read_pkl(config.fact_sen_vocab)
    else:
        if not os.path.isdir(config.save_vocab_path):
            os.makedirs(config.save_vocab_path)
        tra_fact_word_vocab = FactWordVocab([tra_data, dev_data], config)
        pickle.dump(tra_fact_word_vocab, open(config.fact_word_vocab, 'wb'))
        tra_fact_char_vocab = FactCharVocab([tra_data, dev_data], config)
        pickle.dump(tra_fact_char_vocab, open(config.fact_char_vocab, 'wb'))
        if config.read_sen:
            tra_fact_sen_vocab = FactSenVocab([tra_sen_data, dev_sen_data], config)
            pickle.dump(tra_fact_sen_vocab, open(config.fact_sen_vocab, 'wb'))

    model = MyBertModel(config)

    if config.use_cuda:
        model.cuda()

    # train
    train(model, tra_data, dev_data, tra_fact_word_vocab, config)
