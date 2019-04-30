# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function
import tensorflow as tf

from layers import *
from hyperparams import Hyperparams as hp
from data_load_ml import *
from modules import *
from utils import *


class Generator():
    def __init__(self, is_training=True):

        self.user = tf.placeholder(tf.int32, shape=(None,))
        self.item_cand = tf.placeholder(tf.int32, shape=(None, hp.seq_length))
        self.card_idx = tf.placeholder(tf.int32, shape=(None, hp.res_length))
        # self.item_pos = tf.placeholder(tf.int32, shape=(None,))
        # define decoder inputs
        self.decode_target_ids = tf.placeholder(dtype=tf.int32, shape=[hp.batch_size, hp.res_length],
                                                name="decoder_target_ids")  # [batch_size, res_length]
        self.reward = tf.placeholder(dtype=tf.float32, shape=[hp.batch_size],
                                     name="reward")  # [batch_size]

        # Load vocabulary
        user2idx, idx2user = load_user_vocab()
        item2idx, idx2item = load_item_vocab()

        # Encoder
        with tf.variable_scope("encoder"):
            ## Embedding
            # enc_user = [batch_size, hidden_units]
            self.enc_user = embedding(self.user,
                                      vocab_size=len(user2idx),
                                      num_units=hp.hidden_units,
                                      zero_pad=False,
                                      scale=True,
                                      scope="enc_user_embed",
                                      reuse=not is_training)
            # enc_item = [batch_size, seq_len, hidden_units]
            self.enc_item = embedding(self.item_cand,
                                      vocab_size=len(item2idx),
                                      num_units=hp.hidden_units,
                                      zero_pad=False,
                                      scale=True,
                                      scope='enc_item_embed',
                                      reuse=not is_training)
            self.enc = tf.concat([tf.stack(hp.seq_length * [self.enc_user], axis=1), self.enc_item], axis=2)

            ## Dropout
            self.enc = tf.layers.dropout(self.enc,
                                        rate=hp.dropout_rate,
                                        training=tf.convert_to_tensor(is_training))

            if hp.use_mha:
                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc = multihead_attention(queries=self.enc,
                                                        keys=self.enc,
                                                        num_units=hp.hidden_units*2,
                                                        num_heads=hp.num_heads,
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=False)

                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units*2])
            else:
                cell = tf.nn.rnn_cell.GRUCell(num_units=hp.hidden_units * 2)
                outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=self.enc, dtype=tf.float32)
                self.enc = outputs

        # Decoder
        with tf.variable_scope("decoder"):
            dec_cell = LSTMCell(hp.hidden_units*2)

            if hp.num_layers > 1:
                cells = [dec_cell] * hp.num_layers
                dec_cell = MultiRNNCell(cells)
            # ptr sampling
            enc_init_state = trainable_initial_state(hp.batch_size, dec_cell.state_size)
            sampled_logits, sampled_path, _ = ptn_rnn_decoder(
                dec_cell, None,
                self.enc, enc_init_state,
                hp.seq_length, hp.res_length, hp.hidden_units*2,
                hp.num_glimpse, hp.batch_size,
                mode="SAMPLE", reuse=False, beam_size=None)
            # logits: [batch_size, res_length, seq_length]
            self.sampled_logits = tf.identity(sampled_logits, name="sampled_logits")
            # sample_path: [batch_size, res_length]
            self.sampled_path = tf.identity(sampled_path, name="sampled_path")
            self.sampled_result = batch_gather(self.item_cand, self.sampled_path)


            # self.decode_target_ids is placeholder
            decoder_logits, _ = ptn_rnn_decoder(
                dec_cell, self.decode_target_ids,
                self.enc, enc_init_state,
                hp.seq_length, hp.res_length, hp.hidden_units*2,
                hp.num_glimpse, hp.batch_size,
                mode="TRAIN", reuse=True, beam_size=None)
            self.dec_logits = tf.identity(decoder_logits, name="dec_logits")

            supervised_logits, _ = ptn_rnn_decoder(
                dec_cell, self.card_idx,
                self.enc, enc_init_state,
                hp.seq_length, hp.res_length, hp.hidden_units*2,
                hp.num_glimpse, hp.batch_size,
                mode="TRAIN", reuse=True, beam_size=None)
            self.supervised_logits = tf.identity(supervised_logits, name="supervised_logits")

            _, infer_path, _ = ptn_rnn_decoder(
                dec_cell, None,
                self.enc, enc_init_state,
                hp.seq_length, hp.res_length, hp.hidden_units*2,
                hp.num_glimpse, hp.batch_size,
                mode="BEAMSEARCH", reuse=True, beam_size=hp.beam_size)
            self.infer_path = tf.identity(infer_path, name="infer_path")
            self.infer_result = batch_gather(self.item_cand, self.infer_path)

        if is_training:
            # Loss
            # self.y_smoothed = label_smoothing(tf.one_hot(self.decode_target_ids, depth=hp.data_length))
            self.r_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.dec_logits,
                                                                         labels=self.decode_target_ids)

            if hp.schedule_sampling:
                self.s_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.dec_logits,
                                                                             labels=self.card_idx)
            else:
                self.s_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.supervised_logits,
                                                                             labels=self.card_idx)

            # reinforcement
            self.policy_loss = tf.reduce_mean(tf.reduce_sum(self.r_loss, axis=1) * self.reward)

            # supervised loss
            self.supervised_loss = tf.reduce_mean(tf.reduce_sum(self.s_loss, axis=1))

            self.loss = (1.0 - hp.supervised_coe) * self.policy_loss + hp.supervised_coe * self.supervised_loss

            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr_gen, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)


class Discriminator():
    def __init__(self, is_training=True, is_testing=False):
        if is_training and is_testing:
            raise TypeError('is_training and is_testing cannot be both true!')

        if is_training:
            self.user, self.card, self.label, self.num_batch = get_dis_batch_data(is_training=True)
        elif is_testing:
            self.user, self.card, self.label, self.num_batch = get_dis_batch_data(is_training=False)
        else:
            self.user = tf.placeholder(tf.int32, shape=(hp.batch_size,))
            self.card = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.res_length))

        # Load vocabulary
        user2idx, idx2user = load_user_vocab()
        item2idx, idx2item = load_item_vocab()

        ## Embedding
        # enc_user = [batch_size, hidden_units]
        self.enc_user = embedding(self.user,
                                  vocab_size=len(user2idx),
                                  num_units=hp.hidden_units,
                                  zero_pad=False,
                                  scale=True,
                                  scope="enc_user_embed",
                                  reuse= not is_training)
        # enc_card_pos = [batch_size, res_len, hidden_units]
        self.enc_card = embedding(self.card,
                                  vocab_size=len(item2idx),
                                  num_units=hp.hidden_units,
                                  zero_pad=False,
                                  scale=True,
                                  scope='enc_card_embed',
                                  reuse=not is_training)

        ## Dropout
        self.enc_user = tf.layers.dropout(self.enc_user,
                                          rate=hp.dropout_rate,
                                          training=tf.convert_to_tensor(is_training))
        self.enc_card = tf.layers.dropout(self.enc_card,
                                          rate=hp.dropout_rate,
                                          training=tf.convert_to_tensor(is_training))

        self.dis_logits = ctr_dicriminator(self.enc_user, self.enc_card,
                                           hidden_dim=hp.dis_hidden_size)
        self.dis_probs = tf.sigmoid(self.dis_logits)

        self.dis_reward = (self.dis_probs - 0.5) * 2.0

        if is_training or is_testing:
            self.dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,
                                                                                   logits=self.dis_logits))

            self.dis_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_float(tf.greater_equal(self.dis_probs, 0.5)),
                                                               self.label)))

        if is_training:
            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr_dis, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.dis_loss, global_step=self.global_step)