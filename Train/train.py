import time
import numpy as np
import os

from DataProcessing.data_batchiter import create_batch, create_sen_batch
from Model.loss import *
from pytorch_transformers import AdamW, ConstantLRSchedule

from Units.units import decay_learning_rate, data_split, database


def train(model, tra_data, dev_data, tra_word_vocab, config):
    optimizer = AdamW(model.parameters(), lr=config.bert_lr, correct_bias=config.correct_bias, weight_decay=config.weight_decay)

    tra_word_data_iter = create_batch(tra_data, tra_word_vocab, config.batch_size, config, shuffle=False)
    dev_word_data_iter = create_batch(dev_data, tra_word_vocab, config.dev_batch_size, config, shuffle=False)

    random_word_iter = data_split(tra_word_data_iter, config.n_fold)
    tra_word_data_iter, dev_database = database(random_word_iter, config.k, config)

    # Get start!
    global_step = 0

    best_acc = 0
    best_tra_acc = 0

    for epoch in range(0, config.epoch):
        score = 0
        print('\nThe epoch is starting.')
        epoch_start_time = time.time()
        batch_iter = 0
        batch_num = int(len(tra_word_data_iter))
        print('The epoch is :', str(epoch))
        if config.use_lr_decay:
            optimizer = decay_learning_rate(config, optimizer, epoch)
            print("now word_ga lr is {}".format(optimizer.param_groups[0].get("lr")), '\n')
        for word_batch in tra_word_data_iter:
            start_time = time.time()
            model.train()
            batch_size = tra_word_data_iter[0][0].size(0) / 2
            src_premise_matrix, src_hypothesis_matrix, p_mask, h_mask, tag_matrix = word_batch[0], \
                                                                                    word_batch[1], \
                                                                                    word_batch[2], \
                                                                                    word_batch[3], \
                                                                                    word_batch[4]
            logit_a, logit_b = model(src_premise_matrix, src_hypothesis_matrix, p_mask, h_mask)
            loss, correct = tri_loss(logit_a, logit_b, config)
            loss = loss / config.update_every
            loss.backward()
            loss_value = loss.item()
            accuracy = 100.0 * int(correct) / batch_size
            during_time = float(time.time() - start_time)
            print('Step:{}, Epoch:{}, batch_iter:{}, accuracy:{:.4f}({}/{}),'
                  'time:{:.2f}, loss:{:.6f}'.format(global_step, epoch, batch_iter, accuracy, correct, batch_size,
                                                    during_time, loss_value))
            batch_iter += 1

            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                if config.clip_max_norm_use:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            score += correct

            if batch_iter % config.test_interval == 0 or batch_iter == batch_num:
                dev_score = evaluate(model, dev_data, dev_word_data_iter, config)
                if best_acc < dev_score:
                    print('The best dev is' + str(dev_score))
                    best_acc = dev_score
                    if os.path.exists(config.save_model_path):
                        torch.save(model.state_dict(), config.bert_model_pkl)
                    else:
                        os.makedirs(config.save_model_path)
                        torch.save(model.state_dict(), config.bert_model_pkl)
        epoch_time = float(time.time() - epoch_start_time)
        tra_score = 100.0 * score / len(tra_data)
        if tra_score > best_tra_acc:
            best_tra_acc = tra_score
            print('the best_train score is:{}({}/{})'.format(tra_score, score, len(tra_data)))
        print("epoch_time is:", epoch_time)


def evaluate(model, data, word_data_iter, config):
    model.eval()
    get_score = 0
    start_time = time.time()
    for word_batch in word_data_iter:
        start_time = time.time()
        src_premise_matrix, src_hypothesis_matrix, p_mask, h_mask, tag_matrix = word_batch[0], \
                                                                                word_batch[1], \
                                                                                word_batch[2], \
                                                                                word_batch[3], \
                                                                                word_batch[4]
        logit_a, logit_b = model(src_premise_matrix, src_hypothesis_matrix, p_mask, h_mask)
        loss, correct = less_triplet_loss(logit_a, logit_b, config)
        get_score += correct
    eval_score = 100.0 * get_score / len(data)
    during_time = float(time.time() - start_time)
    print('the dev score is:{}({}/{})'.format(eval_score, get_score, len(data)))
    print('spent time is:{:.4f}'.format(during_time))
    return eval_score
