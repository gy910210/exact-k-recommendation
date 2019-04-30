# -*- coding: utf-8 -*-
#/usr/bin/python2

class Hyperparams:
    '''Hyperparameters'''
    # data
    data_set = 'ml100k-10c'

    gen_data_train = 'data/' + data_set + '/rerank_data_10c_train.txt'
    gen_data_test = 'data/' + data_set + '/rerank_data_10c_test.txt'

    dis_data_train = 'data/' + data_set + '/dis_data_10c_train.txt'
    dis_data_test = 'data/' + data_set + '/dis_data_10c_test.txt'

    user_ids_file = 'data/' + data_set + '/user_ids.txt'
    item_ids_file = 'data/' + data_set + '/item_ids.txt'

    # training
    batch_size = 32 # alias = N
    num_glimpse = 1
    beam_size = 3
    num_layers = 1 # rnn layer num
    seq_length = 50 # encoder length
    res_length = 10
    lr_dis = 0.001 # learning rate.
    lr_gen = 0.001
    logdir = 'logdir' # log directory

    print_per_step = 10
    test_per_step = 10

    gen_num_epochs = 5
    dis_num_epochs = 1

    # hill climbling
    is_hill_climbing = True
    num_hill_climb = 32 # sample大小，目前仅支持batch_size的倍数
    top_k_candidate = 3 # top k candidate的大小

    # model
    hidden_units = 16 # alias = C, for embedding size and rnn cell
    dis_hidden_size = 128 # for discriminator
    num_blocks = 2 # number of encoder/decoder blocks
    num_heads = 2
    dropout_rate = 0.1

    supervised_coe = 1.0
    schedule_sampling = True
    use_mha = False
    use_dis_reward = True

    # log print
    gen_train_log_path = 'gen_train_log.txt'
    gen_test_log_path = 'gen_test_log.txt'

    dis_train_log_path = 'dis_train_log.txt'
    dis_test_log_path = 'dis_test_log.txt'