# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function
import tensorflow as tf

from layers import *
from hyperparams import Hyperparams as hp
from data_load_ml import *
from modules import *
import os, codecs
from tqdm import tqdm
from utils import *
from model import Generator, Discriminator


if __name__ == '__main__':
    # load gen data
    gen_user, gen_card, gen_card_idx, gen_item_cand, gen_item_pos, gen_num_batch \
        = get_gen_batch_data(is_training=True)
    gen_user_test, gen_card_test, _, gen_item_cand_test, gen_item_pos_test, gen_num_batch_test \
        = get_gen_batch_data(is_training=False)

    # Construct graph
    with tf.name_scope('Generator'):
        g = Generator(is_training=True)
    print(len(tf.get_variable_scope().global_variables()))
    with tf.name_scope('Discriminator'):
        d = Discriminator(is_training=True, is_testing=False)
    print(len(tf.get_variable_scope().global_variables()))

    tf.get_variable_scope().reuse_variables()
    with tf.name_scope('DiscriminatorInfer'):
        d_infer = Discriminator(is_training=False, is_testing=False)
    with tf.name_scope('DiscriminatorTest'):
        d_test = Discriminator(is_training=False, is_testing=True)
    with tf.name_scope('GeneratorInfer'):
        g_infer = Generator(is_training=False)

    print("Graph loaded")

    # Load vocabulary
    user2idx, idx2user = load_user_vocab()
    item2idx, idx2item = load_item_vocab()

    # log file init
    gen_train_log = open(os.path.join(hp.logdir, hp.gen_train_log_path), 'w')
    gen_train_log.write('step\tgen_reward\tprecision@4\tprecision\n')
    gen_test_log = open(os.path.join(hp.logdir, hp.gen_test_log_path), 'w')
    gen_test_log.write('step\tgen_reward\tprecision@4\tprecision\n')
    dis_train_log = open(os.path.join(hp.logdir, hp.dis_train_log_path), 'w')
    dis_train_log.write('step\tdis_loss\tdis_acc\n')
    dis_test_log = open(os.path.join(hp.logdir, hp.dis_test_log_path), 'w')
    dis_test_log.write('step\tdis_loss\tdis_acc\n')

    # Start session
    sv = tf.train.Supervisor(is_chief= True,
                             summary_op=None,
                             logdir=hp.logdir,
                             save_model_secs=0)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.95,
        allow_growth=True)  # seems to be not working
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=gpu_options)

    with sv.managed_session(config=sess_config) as sess:
        print('Discriminator training start!')

        dis_acc_best = 0.0
        dis_loss_total, dis_acc_total = 0.0, 0.0
        for epoch in range(1, hp.dis_num_epochs + 1):
            if sv.should_stop():
                break
            print('Discriminator epoch: ', epoch)

            # for step in tqdm(range(d.num_batch), total=d.num_batch, ncols=70, leave=False, unit='b'):
            for step in range(d.num_batch):
                gs_dis = sess.run(d.global_step)
                _, dis_loss, dis_acc = sess.run([d.train_op, d.dis_loss, d.dis_acc])
                dis_loss_total += dis_loss
                dis_acc_total += dis_acc

                ## print
                if (gs_dis + 1) % hp.print_per_step == 0:
                    print('gs_dis: {}, dis_loss_train: {}, dis_acc_train: {}'.format(
                        (gs_dis + 1),
                        dis_loss_total / (1.0 * (gs_dis + 1)),
                        dis_acc_total / (1.0 * (gs_dis + 1))))
                    dis_train_log.write('{}\t{}\t{}\n'.format(
                        (gs_dis + 1),
                        dis_loss_total / (1.0 * (gs_dis + 1)),
                        dis_acc_total / (1.0 * (gs_dis + 1))))
                    dis_train_log.flush()

                ## test
                if (gs_dis + 1) % hp.test_per_step == 0:
                    dis_loss_test, dis_acc_test = 0.0, 0.0
                    for _ in range(d_test.num_batch):
                        dis_loss, dis_acc = sess.run([d_test.dis_loss, d_test.dis_acc])
                        dis_loss_test += dis_loss
                        dis_acc_test += dis_acc
                    dis_loss_test /= (1.0 * d_test.num_batch)
                    dis_acc_test /= (1.0 * d_test.num_batch)
                    print('gs_dis: {}, dis_loss_test: {}, dis_acc_test: {}'.format(
                        (gs_dis + 1), dis_loss_test, dis_acc_test))
                    dis_test_log.write('{}\t{}\t{}\n'.format(
                        (gs_dis + 1), dis_loss_test, dis_acc_test))
                    dis_test_log.flush()

                    if dis_acc_test > dis_acc_best:
                        dis_acc_best = dis_acc_test
                        print('dis_acc_best: ', dis_acc_best)
                        sv.saver.save(sess, hp.logdir + '/model/best_model')

        print('Discriminator training done!')

        sv.saver.restore(sess, hp.logdir + '/model/best_model')

        print('Generator training start!')
        # 记录sample到的最好的结果
        memory_reward = {}
        memory_card_idx = {}
        memory_card = {}

        precision_at_4_best, precision_best = 0.0, 0.0
        reward_total, precision_at_4_total, precision_total = 0.0, 0.0, 0.0
        for epoch in range(1, hp.gen_num_epochs + 1):
            if sv.should_stop():
                break
            print('Generator epoch: ', epoch)

            # for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
            for step in range(gen_num_batch):
                user, card, card_idx, item_cand, item_pos = \
                    sess.run([gen_user, gen_card, gen_card_idx, gen_item_cand, gen_item_pos])

                if hp.is_hill_climbing:
                    samples = []
                    for i in range(hp.batch_size):
                        user_i = np.tile(user[i], (hp.num_hill_climb))
                        item_cand_i = np.tile(item_cand[i], (hp.num_hill_climb, 1))
                        hill_sampled_card_idx, hill_sampled_card = sess.run([g.sampled_path, g.sampled_result],
                                                                            feed_dict={g.user: user_i,
                                                                                       g.item_cand: item_cand_i})
                        hill_reward = sess.run(d_infer.dis_reward,
                                               feed_dict={d_infer.card: hill_sampled_card,
                                                          d_infer.user: user_i})
                        sorted_list = sorted(list(zip(hill_sampled_card, hill_sampled_card_idx, hill_reward)),
                                             key=lambda x: x[2], reverse=True)
                        samples.append(sorted_list[np.random.choice(hp.top_k_candidate)])

                        if user[i] not in memory_reward:
                            memory_reward[user[i]] = sorted_list[0][2]
                            memory_card_idx[user[i]] = sorted_list[0][1]
                            memory_card[user[i]] = sorted_list[0][0]
                        else:
                            if memory_reward[user[i]] > sorted_list[0][2]:
                                memory_reward[user[i]] = sorted_list[0][2]
                                memory_card_idx[user[i]] = sorted_list[0][1]
                                memory_card[user[i]] = sorted_list[0][0]
                    (sampled_card, sampled_card_idx, reward) = zip(*samples)
                else:
                    # sample
                    sampled_card_idx, sampled_card = sess.run([g.sampled_path, g.sampled_result],
                                                              feed_dict={g.user: user, g.item_cand: item_cand})

                    if hp.use_dis_reward:
                        reward = sess.run(d_infer.dis_reward,
                                          feed_dict={d_infer.card: sampled_card, d_infer.user: user})
                    else:
                        reward = []
                        for i in range(len(sampled_card)):
                            if item_pos[i] in set(sampled_card[i]):
                                reward.append(1.0)
                            else:
                                reward.append(-1.0)

                # train
                sess.run(g.train_op, feed_dict={g.decode_target_ids: sampled_card_idx,
                                                g.reward: reward,
                                                g.item_cand: item_cand,
                                                g.user: user,
                                                g.card_idx: card_idx})
                gs_gen = sess.run(g.global_step)
                reward_total += np.mean(reward)

                # beamsearch
                beam_card = sess.run(g_infer.infer_result,
                                     feed_dict={g_infer.item_cand: item_cand,
                                                g_infer.user: user})
                precision_at_4_total += precision_at_4(beam_card, item_pos)
                precision_total += precision(beam_card, card)

                ## print
                if (gs_gen + 1) % hp.print_per_step == 0:
                    print('gs_gen: {}, gen_reward_train: {}, precision@4_train: {}, precision_train: {}'.format(
                        (gs_gen + 1),
                        reward_total / (1.0 * (gs_gen + 1)),
                        precision_at_4_total / (1.0 * (gs_gen + 1)),
                        precision_total / (1.0 * (gs_gen + 1))))
                    gen_train_log.write('{}\t{}\t{}\t{}\n'.format(
                        (gs_gen + 1),
                        reward_total / (1.0 * (gs_gen + 1)),
                        precision_at_4_total / (1.0 * (gs_gen + 1)),
                        precision_total / (1.0 * (gs_gen + 1))))
                    gen_train_log.flush()

                ## test
                if (gs_gen + 1) % hp.test_per_step == 0:
                    precision_at_4_test, precision_test, reward_test = 0.0, 0.0, 0.0
                    for _ in range(gen_num_batch_test):
                        user_test, card_test, item_cand_test, item_pos_test \
                            = sess.run([gen_user_test, gen_card_test, gen_item_cand_test, gen_item_pos_test])
                        beam_card_test = sess.run(g_infer.infer_result,
                                                  feed_dict={g_infer.item_cand: item_cand_test,
                                                             g_infer.user: user_test})
                        precision_at_4_test += precision_at_4(beam_card_test, item_pos_test)
                        precision_test += precision(beam_card_test, card_test)
                        reward = sess.run(d_infer.dis_reward,
                                          feed_dict={d_infer.card: beam_card_test,
                                                     d_infer.user: user_test})
                        reward_test += np.mean(reward)

                    reward_test /= (1.0 * gen_num_batch_test)
                    precision_at_4_test /= (1.0 * gen_num_batch_test)
                    precision_test /= (1.0 * gen_num_batch_test)
                    print('gs_gen: {}, gen_reward_test: {}, precision@4_test: {}, precision_test: {}'.format(
                        (gs_gen + 1), reward_test, precision_at_4_test, precision_test))
                    gen_test_log.write('{}\t{}\t{}\t{}\n'.format(
                        (gs_gen + 1), reward_test, precision_at_4_test, precision_test))
                    gen_test_log.flush()

                    if precision_at_4_test > precision_at_4_best:
                        precision_at_4_best = precision_at_4_test
                        precision_best = precision_test
                        print('precision_at_4_best: ', precision_at_4_best,
                              'precision_best: ', precision_best)
                        sv.saver.save(sess, hp.logdir + '/model/best_model')

        print('Generator training done!')

    print("Done")

    gen_train_log.close()
    gen_test_log.close()
    dis_train_log.close()
    dis_test_log.close()