from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import random


def load_user_vocab():
    user_ids = [line.strip() for line in open(hp.user_ids_file, 'r').read().splitlines()]
    user2idx = {int(user): idx for idx, user in enumerate(user_ids)}
    idx2user = {idx: int(user) for idx, user in enumerate(user_ids)}
    return user2idx, idx2user


def load_item_vocab():
    item_ids = [line.strip() for line in open(hp.item_ids_file, 'r').read().splitlines()]
    item2idx = {int(item): idx for idx, item in enumerate(item_ids)}
    idx2item = {idx: int(item) for idx, item in enumerate(item_ids)}
    return item2idx, idx2item

##########################################

def load_gen_data(file_path):
    user2idx, _ = load_user_vocab()
    item2idx, _ = load_item_vocab()

    USER, CARD, CARD_IDX, ITEM_CAND, ITEM_POS = [], [], [], [], []
    with open(file_path, 'r') as fin:
        for line in fin:
            strs = line.strip().split('\t')
            USER.append(user2idx[int(strs[0])])
            card_ = [item2idx[int(x)] for x in strs[1].split(',')]
            CARD.append(card_)
            item_cand_ = sorted([item2idx[int(x)] for x in strs[2].split(',')])
            ITEM_CAND.append(item_cand_) # sorted
            ITEM_POS.append(card_[0])

            item_cand_idx_map = {}
            for idx, item in enumerate(item_cand_):
                item_cand_idx_map[item] = idx
            card_idx_ = [item_cand_idx_map[item] for item in card_]
            CARD_IDX.append(card_idx_)
            '''
            tmp = set(strs[2].split(','))
            tmp.remove(strs[1].split(',')[0])
            tmp = list(tmp)
            random.shuffle(tmp)
            ITEM_CAND_NEG.append([item2idx[int(x)] for x in tmp])
            '''

    return USER, CARD, CARD_IDX, ITEM_CAND, ITEM_POS


def get_gen_batch_data(is_training=True):
    # Load data
    if is_training:
        USER, CARD, CARD_IDX, ITEM_CAND, ITEM_POS = load_gen_data(hp.gen_data_train)
        batch_size = hp.batch_size
        print('Load generator training data done!')
    else:
        USER, CARD, CARD_IDX, ITEM_CAND, ITEM_POS = load_gen_data(hp.gen_data_test)
        batch_size = hp.batch_size
        print('Load generator testing data done!')

    # calc total batch count
    num_batch = len(USER) // batch_size

    # Convert to tensor
    USER = tf.convert_to_tensor(USER, tf.int32) # [batch_size]
    CARD = tf.convert_to_tensor(CARD, tf.int32) # [batch_size, 4]
    CARD_IDX = tf.convert_to_tensor(CARD_IDX, tf.int32) # [batch_size, 4]
    ITEM_CAND = tf.convert_to_tensor(ITEM_CAND, tf.int32) # [batch_size, 20]
    ITEM_POS = tf.convert_to_tensor(ITEM_POS, tf.int32) # [batch_size]
    # ITEM_CAND_NEG = tf.convert_to_tensor(ITEM_CAND_NEG, tf.int32) # [batch_size, 19]

    # Create Queues
    input_queues = tf.train.slice_input_producer([USER, CARD, CARD_IDX, ITEM_CAND, ITEM_POS])

    # create batch queues
    user, card, card_idx, item_cand, item_pos = \
        tf.train.shuffle_batch(input_queues,
                               num_threads=8,
                               batch_size=batch_size,
                               capacity=batch_size * 64,
                               min_after_dequeue=batch_size * 32,
                               allow_smaller_final_batch=False)

    # card_neg = tf.random_crop(item_cand_neg, size=[hp.batch_size, hp.res_length])
    return user, card, card_idx, item_cand, item_pos, num_batch

#####################################

def load_dis_data(file_path):
    user2idx, _ = load_user_vocab()
    item2idx, _ = load_item_vocab()

    USER, CARD, LABEL = [], [], []
    with open(file_path, 'r') as fin:
        for line in fin:
            strs = line.strip().split('\t')
            USER.append(user2idx[int(strs[0])])
            card = [item2idx[int(x)] for x in strs[1].split(',')]
            random.shuffle(card)  # shuffled
            CARD.append(card)
            LABEL.append(float(strs[2]))

    return USER, CARD, LABEL


def get_dis_batch_data(is_training=True):
    # Load data
    if is_training:
        USER, CARD, LABEL = load_dis_data(hp.dis_data_train)
        batch_size = hp.batch_size
        print('Load discriminator training data done!')
    else:
        USER, CARD, LABEL = load_dis_data(hp.dis_data_test)
        batch_size = hp.batch_size
        print('Load discriminator testing data done!')

    # calc total batch count
    num_batch = len(USER) // batch_size

    # Convert to tensor
    USER = tf.convert_to_tensor(USER, tf.int32)  # [batch_size]
    CARD = tf.convert_to_tensor(CARD, tf.int32)  # [batch_size, 4]
    LABEL = tf.convert_to_tensor(LABEL, tf.float32)  # [batch_size]

    # Create Queues
    input_queues = tf.train.slice_input_producer([USER, CARD, LABEL])

    # create batch queues
    user, card, label = \
        tf.train.shuffle_batch(input_queues,
                               num_threads=8,
                               batch_size=batch_size,
                               capacity=batch_size * 64,
                               min_after_dequeue=batch_size * 32,
                               allow_smaller_final_batch=False)

    return user, card, label, num_batch


if __name__ == "__main__":
    user, card, card_idx, item_cand, item_pos, num_batch = get_gen_batch_data(is_training=True)
    print(user)
    print(card)
    print(card_idx)
    print(item_cand)
    print(item_pos)
    print(str(num_batch))

    user, card, label, num_batch = get_dis_batch_data(is_training=True)
    print(user)
    print(card)
    print(label)