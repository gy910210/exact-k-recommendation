# encoding: UTF-8


import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.python.framework import tensor_util
# from tensorflow.contrib\
from tensorflow.python.util import nest
import numpy as np
from utils import index_matrix_to_pairs_fn
from hyperparams import Hyperparams as hp

try:
    from tensorflow.contrib.layers.python.layers import utils  # 1.0.0
except:
    from tensorflow.contrib.layers import utils

smart_cond = utils.smart_cond

try:
    LSTMCell = rnn.LSTMCell  # 1.0.0
    MultiRNNCell = rnn.MultiRNNCell
    # dynamic_rnn_decoder = seq2seq.dynamic_rnn_decoder
    # simple_decoder_fn_train = seq2seq.simple_decoder_fn_train
except:
    LSTMCell = tf.contrib.rnn.LSTMCell
    MultiRNNCell = tf.contrib.rnn.MultiRNNCell
    # dynamic_rnn_decoder = tf.contrib.seq2seq.dynamic_rnn_decoder
    # simple_decoder_fn_train = tf.contrib.seq2seq.simple_decoder_fn_train

def ptn_rnn_decoder(cell,
                    decoder_target_ids,
                    enc_outputs,
                    enc_final_states,
                    seq_length,
                    res_length,
                    hidden_dim,
                    num_glimpse,
                    batch_size,
                    initializer=None,
                    mode="SAMPLE",
                    reuse=False,
                    beam_size=None):
    """
    :param cell:
    :param decoder_target_ids:
    :param enc_outputs:
    :param enc_final_states:
    :param seq_length:
    :param hidden_dim:
    :param num_glimpse:
    :param batch_size:
    :param initializer:
    :param mode: SAMPLE/GREEDY/BEAMSEARCH/TRAIN, if TRAIN, decoder_input_ids shouldn't be none
    :param reuse:
    :param beam_size: a positive int if mode="BEAMSEARCH"
    :return: [logits, sampled_ids, final_state], shape: [batch_size, seq_len, data_len], [batch, seq_len], state_size
    """
    with tf.variable_scope("decoder_rnn") as scope:
        if reuse:
            scope.reuse_variables()

        first_decoder_input = trainable_initial_state(
            batch_size, hidden_dim, initializer=None, name="first_decoder_input")

        # 多次decode计算attention时，计算encoder*参数只计算一次
        enc_refs = {}
        dec_qs = {}
        # 存储已经decoder的序列,用于计算intra-attention
        output_ref = []
        index_matrix_to_pairs = index_matrix_to_pairs_fn(batch_size, seq_length)

        def intra_attention(bef, query, scope="intra_attention"):
            """
             :param bef: decoder阶段的已输出序列[batch,decoder_len,hidden_dim] decoder_len为目前decoder的长度
             :param query: decoder的输出
             :return: intra_attention:[batch,hidden_dim]
             """
            with tf.variable_scope(scope) as scope:
                W_b = tf.get_variable(
                    "W_b", [hidden_dim, hidden_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                v_dec = tf.get_variable(
                    "v_dec", [hidden_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                W_bef = tf.get_variable(
                    "W_bef", [1, hidden_dim, hidden_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                bias_dec = tf.get_variable(
                    "bias_dec", [hidden_dim],
                    initializer=tf.zeros_initializer)
                if len(bef) <= 1:
                    if len(bef) == 0:
                        return tf.zeros([batch_size, hidden_dim])
                    else:
                        return bef[0]
                else:
                    bef = tf.stack(bef, axis=1)
                    # bef_rs = tf.reduce_sum(bef_s,axis=[2])

                decoded_bef = tf.nn.conv1d(bef, W_bef, 1, "VALID",
                                           name="decoded_bef")  # [batch, decoder_len, hidden_dim]
                decoded_query = tf.expand_dims(tf.matmul(query, W_b, name="decoded_query"), 1)  # [batch, 1, hidden_dim]
                scores = tf.reduce_sum(v_dec * tf.tanh(decoded_bef + decoded_query + bias_dec),
                                       [-1])  # [batch, decoder_len]
                p1 = tf.nn.softmax(scores)
                aligments1 = tf.expand_dims(p1, axis=2)
                return tf.reduce_sum(aligments1 * bef, axis=[1])

        def attention(ref, query, dec_ref, with_softmax, scope="attention"):
            """
             :param ref: [batch, seq_length, hidden_dim]  encoder阶段的序列
             :param dec_ref: [batch,hidden_dim] decoder阶段的intra-decoder-attention的结果
             :param query: [batch, hidden_dim]     decoder的输出
             :return  attention: [batch, seq_length]
             """
            with tf.variable_scope(scope) as scope:
                W_q = tf.get_variable(
                    "W_q", [hidden_dim, hidden_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                W_dec = tf.get_variable(
                    "W_dec", [hidden_dim, hidden_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                v = tf.get_variable(
                    "v", [hidden_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                bias = tf.get_variable(
                    "bias", [hidden_dim],
                    initializer=tf.zeros_initializer)

                enc_ref_key = (ref.name, scope.name)
                if enc_ref_key not in enc_refs:
                    W_ref = tf.get_variable("W_ref", [1, hidden_dim, hidden_dim],
                                            initializer=tf.contrib.layers.xavier_initializer())
                    enc_refs[enc_ref_key] = tf.nn.conv1d(ref, W_ref, 1, "VALID",
                                                         name="encoded_ref")  # [batch, data_len, hidden_dim]
                encoded_ref = enc_refs[enc_ref_key]
                encoded_query = tf.expand_dims(tf.matmul(query, W_q, name="encoded_query"), 1)  # [batch, 1, hidden_dim]
                decoded_ref = tf.expand_dims(tf.matmul(dec_ref, W_dec, name="decoded_ref"), 1)  # [batch, 1, hidden_dim]
                scores = tf.reduce_sum(v * tf.tanh(encoded_ref + encoded_query + decoded_ref + bias),
                                       [-1])  # [batch, data_len]

                if with_softmax:
                    return tf.nn.softmax(scores)
                else:
                    return scores

        def glimpse(ref, query, dec_ref, scope="glimpse"):
            """
            :param ref: [batch, seq_length, hidden_dim]
            :param query: [batch, hidden_dim]
            :param dec_ref: [batch,hidden_dim] decoder阶段的intra-decoder-attention的结果
            :return g: [batch, hidden_dim]
            """
            p = attention(ref, query, dec_ref, with_softmax=True, scope=scope)
            alignments = tf.expand_dims(p, axis=2)  # [batch, data_len, 1]
            return tf.reduce_sum(alignments * ref, axis=[1])

        def output_fn(ref, query, dec_ref, num_glimpse):
            """
            :param ref: [batch, seq_length, hidden_dim]
            :param query: [batch, hidden_dim]
            :param dec_ref: [batch,hidden_dim] decoder阶段的intra-decoder-attention的结果
            :param num_glimpse: 1
            :return: [batch_size, seq_length]
            """

            for idx in range(num_glimpse):
                query = glimpse(ref, query, dec_ref, "glimpse_{}".format(idx))
            return attention(ref, query, dec_ref, with_softmax=False, scope="attention")

        def input_fn(input_idx):
            """
            turn input_idx to encoder_output vector
            :param input_idx: [batch_size] or [batch_size, seq_length]
            :return: [batch_size, hidden_dim] or [batch_size, seq_length, hidden_dim]
            """
            # enc_outputs: [batch_size, seq_length, hidden_dim]
            # input_index_pairs: [batch_size, 2]
            # input_index_pairs = tf.stop_gradient(tf.stack(
            #     [tf.range(tf.shape(input_idx)[0], dtype=tf.int32), input_idx], axis=1))
            input_index_pairs = tf.stop_gradient(index_matrix_to_pairs(input_idx))
            return tf.gather_nd(enc_outputs, input_index_pairs)

        def random_sample_from_logits(logits):
            sampled_idx = tf.cast(tf.multinomial(logits=logits, num_samples=1), dtype='int32')  # [batch_size,1]
            sampled_idx = tf.reshape(sampled_idx, [batch_size])  # [batch_size]
            return sampled_idx

        def greedy_sample_from_logits(logits):
            # logits: [batch, seq_length]
            return tf.cast(tf.argmax(logits, 1), tf.int32)

        def call_cell(input_idx, state, point_mask):
            """
              call lstm_cell and compute attention and intra-attention
            :param input_idx: [batch]
            :param state:
            :param point_mask: [batch, seq_length]
            :return: [batch_size, seq_length]
            """
            if input_idx is not None:
                _input = input_fn(input_idx)  # [batch_size, hidden_dim]
            else:
                _input = first_decoder_input

            cell_output, new_state = cell(_input, state)
            # 先计算 intra-decoder-attention
            intra_dec = intra_attention(output_ref, cell_output)  # [batch_size, hidden_dim]
            output_ref.append(cell_output)
            logits = output_fn(enc_outputs, cell_output, intra_dec, num_glimpse)  # [batch_size, data_len]
            if point_mask is not None:
                max_logit = tf.reduce_max(logits)
                min_logit = tf.reduce_min(logits)
                # 确保先前选过的点不再选，设置logit为min_logit-9999，并阻止梯度回传。
                logits = tf.minimum(logits,
                                    tf.stop_gradient(max_logit + 1 + tf.cast(point_mask, dtype=tf.float32) * (
                                    min_logit - 10000 - max_logit)))
            return logits, new_state

        def update_mask(output_idx, old_mask):
            new_mask_inc = tf.one_hot(output_idx, depth=seq_length, dtype='int32')
            new_mask = tf.stop_gradient(old_mask + new_mask_inc)
            return new_mask

        # logits: [batch_size, data_len]
        logits, state = call_cell(input_idx=None, state=enc_final_states, point_mask=None)  # [batch_size, data_len]
        scope.reuse_variables()
        output_logits = [logits]
        point_mask = tf.zeros([batch_size, seq_length], dtype=tf.int32)

        if (mode in ['SAMPLE', "GREEDY"]):
            if mode == "SAMPLE":
                sample_fn = random_sample_from_logits
            elif mode == "GREEDY":
                sample_fn = greedy_sample_from_logits
            else:
                raise NotImplementedError("invalid mode: %s. Available modes: [SAMPLE, GREEDY]" % mode)
            output_idx = sample_fn(logits)  # [batch_size]
            output_idxs = [output_idx]
            point_mask = update_mask(output_idx, point_mask)

            for i in range(1, res_length):
                logits, state = call_cell(output_idx, state, point_mask)  # [batch_size, data_len]
                output_logits.append(logits)
                output_idx = sample_fn(logits)  # [batch_size]
                point_mask = update_mask(output_idx, point_mask)
                output_idxs.append(output_idx)
            return tf.stack(output_logits, axis=1), tf.stack(output_idxs, axis=1), state
        elif mode == "TRAIN":
            output_idxs = tf.unstack(decoder_target_ids, axis=1)
            output_idx = output_idxs[0]  # [batch_size]
            point_mask = update_mask(output_idx, point_mask)

            for i in range(1, res_length):
                logits, state = call_cell(output_idx, state, point_mask)  # [batch_size, data_len]
                output_logits.append(logits)
                output_idx = output_idxs[i]  # [batch_size]
                point_mask = update_mask(output_idx, point_mask)
            return tf.stack(output_logits, axis=1), state
        elif mode == "BEAMSEARCH":
            index_matrix_to_beampairs = index_matrix_to_pairs_fn(batch_size, beam_size)

            def top_k(acum_logits, logits):
                """
                :param acum_logits: [batch] * beam_size
                :param logits: [batch, len] * beam_size
                :return:
                  new_acum_logits [batch] * beam_size,
                  last_beam_id [batch, beam_size], sample_id [batch, beam_size]
                """
                # local_acum_logits: [batch, len*beam_size]
                candicate_size = len(logits)
                local_acum_logits = logits
                if accum_logits is not None:
                    local_acum_logits = [tf.reshape(acum_logits[ik], [-1, 1]) + logits[ik]
                                         for ik in range(candicate_size)]
                # local_acum_logits: [batch, len]*candicate_size -> [batch, len*candicate_size]
                local_acum_logits = tf.concat(local_acum_logits, axis=1)
                # local_acum_logits:[batch, len * candicate_size] -> [batch, beam_size]
                # local_id:[batch, beam_size] \in range(len*candicate_size)
                local_acum_logits, local_id = tf.nn.top_k(local_acum_logits, beam_size)
                last_beam_id = local_id // seq_length
                last_beam_id = index_matrix_to_beampairs(last_beam_id)  # [batch, beam_size, 2]
                sample_id = local_id % seq_length
                new_acum_logits = tf.unstack(local_acum_logits, axis=1)  # [batch] * beam_size
                return new_acum_logits, last_beam_id, sample_id

            def beam_select(inputs_l, beam_id):
                """
                :param input_l: list of tensors, len(input_l) = k
                :param beam_id: [batch, k, 2]
                :return: output_l, list of tensors, len = k
                """

                def _select(input_l):
                    input_l = tf.stack(input_l, axis=1)  # [batch, beam_size, ...]
                    output_l = tf.gather_nd(input_l, beam_id)  # [batch, beam_size, ...]
                    output_l = tf.unstack(output_l, axis=1)
                    return output_l

                # [state, state] -> [(h,c),(h,c)] -> [[h,h,h], [c,c,c]]
                inputs_ta_flat = zip(*[nest.flatten(input_l) for input_l in inputs_l])
                # [[h,h,h], [c,c,c]] -(beam select)> [[h,h,h], [c,c,c]]
                outputs_ta_flat = [_select(input_ta) for input_ta in inputs_ta_flat]
                # [[h,h,h], [c,c,c]] -> [(h,c),(h,c)] -> [state, state]
                outputs_l = [nest.pack_sequence_as(inputs_l[0], output_ta_flat)
                             for output_ta_flat in zip(*outputs_ta_flat)]
                return outputs_l

            def beam_sample(accum_logits, logits, point_mask, state, pre_output_idxs):
                # sample top_k, last_bema_id:[batch,beam_size], output_idx:[batch,beam_size]
                accum_logits, last_beam_id, output_idx = top_k(accum_logits,
                                                               logits)  # [batch, beam_size], 前面那个beam path, 后面哪个节点
                state = beam_select(state, last_beam_id)
                point_mask = beam_select(point_mask, last_beam_id)
                output_idx = tf.unstack(output_idx, axis=1)  # [batch] * beam_size
                point_mask = [update_mask(output_idx[i], point_mask[i]) for i in range(beam_size)]

                l_output_idx = [tf.expand_dims(t, axis=1)  # [batch, 1] * beam_size
                                for t in output_idx]
                if pre_output_idxs is not None:
                    pre_output_idxs = beam_select(pre_output_idxs, last_beam_id)
                    output_idxs = map(lambda ts: tf.concat(ts, axis=1), zip(pre_output_idxs, l_output_idx))
                else:
                    output_idxs = l_output_idx
                return accum_logits, point_mask, state, output_idx, output_idxs

            # initial setting
            state = [state] * beam_size  # [batch, state_size] * beam_size
            point_mask = [point_mask] * beam_size  # [batch, data_len] * beam_size
            # logits -> log pi
            logits = logits - tf.reduce_logsumexp(logits, axis=1, keep_dims=True)
            logits = [logits] * beam_size  # [batch, data_len] * beam_size
            accum_logits = [tf.zeros([batch_size])] * beam_size

            accum_logits, point_mask, state, output_idx, output_idxs = \
                beam_sample(accum_logits, logits, point_mask, state, None)
            for i in range(1, res_length):
                logits, state = zip(*[call_cell(output_idx[ik], state[ik], point_mask[ik])  # [batch_size, data_len]
                                      for ik in range(beam_size)])
                # logits -> log pi
                logits = [logit_ - tf.reduce_logsumexp(logit_, axis=1, keep_dims=True) for logit_ in logits]
                accum_logits, point_mask, state, output_idx, output_idxs = \
                    beam_sample(accum_logits, logits, point_mask, state, output_idxs)
            return accum_logits[0], output_idxs[0], state[0]
        else:
            raise NotImplementedError("unknown mode: %s. Available modes: [SAMPLE, TRAIN, GREEDY, BEAMSEARCH]" % mode)


def trainable_initial_state(batch_size,
                            state_size,
                            initializer=None,
                            name="initial_state"):
    flat_state_size = nest.flatten(state_size)  # Returns a flat sequence from a given nested structure.

    if not initializer:
        flat_initializer = tuple(tf.zeros_initializer for _ in flat_state_size)
    else:
        flat_initializer = tuple(tf.zeros_initializer for initializer in flat_state_size)

    names = ["{}_{}".format(name, i) for i in xrange(len(flat_state_size))]
    tiled_states = []

    # tiled_ta = tf.ones(shape=[batch_size])
    for name, size, init in zip(names, flat_state_size, flat_initializer):
        shape_with_batch_dim = [1, size]
        initial_state_variable = tf.get_variable(
            name, shape=shape_with_batch_dim, initializer=init())

        # tf.multiply(tiled_ta, initial_state_variable, name=(name + "_tiled"))
        tiled_state = tf.tile(initial_state_variable,
                              [batch_size, 1], name=(name + "_tiled"))
        tiled_states.append(tiled_state)

    return nest.pack_sequence_as(structure=state_size,
                                 flat_sequence=tiled_states)



def ctr_dicriminator(user, card, hidden_dim):
    '''
    :param user: [batch_size, user_embedding]
    :param card: [batch_size, res_len, item_embedding]
    :param hidden_dim: dnn hidden dimension
    :return: logit for ctr
    '''
    with tf.variable_scope("ctr_dicriminator"):

        batch_size = user.get_shape()[0].value
        if batch_size is None:
            batch_size = tf.shape(user)[0]

        # user_flat: [batch_size, res_len, user_embedding]
        user_flat = tf.stack(hp.res_length * [user], axis=1)

        cross_feature = tf.reduce_sum(tf.multiply(user_flat, card), axis=2)
        cross_feature = tf.reshape(cross_feature, shape=[batch_size, -1])

        card_feature = tf.reshape(card, shape=[batch_size, -1])

        feature = tf.concat([user, card_feature, cross_feature], axis=1)
        feature = tf.layers.dense(feature, hidden_dim, activation=tf.nn.relu)
        logits = tf.layers.dense(feature, 1, activation=None)
        logits = tf.squeeze(logits, axis=[1])
        return logits