#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import pickle
import glob
from pathlib import Path
from os.path import dirname, abspath


# tf.flags.DEFINE_string("checkpoint_dir", "runs/save_dir_for_test/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")


FLAGS = tf.flags.FLAGS

local_dir = abspath(dirname(__file__))
os.chdir(local_dir)

def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")

    with open('./metadata.pkl', 'rb') as mt:
        metadata = pickle.load(mt)

    maxlen = metadata['maxlen']
    vocab_size = metadata['vocab_size']
    num_classes = metadata['num_classes']
    weights = metadata['weights']

    print("Vocabulary Size: {:d}".format(vocab_size))
    print("Number of classes: {:d}".format(num_classes))
    print("Max length: {:d}".format(maxlen))
    print("Weights: ")
    print(weights)

    return maxlen, vocab_size, num_classes, weights



def retrain(maxlen, vocab_size, num_classes, weights):

    # print(glob.glob('runs/*'))
    # p = Path('runs/')
    # lof = glob.glob('runs/*')
    lof = os.listdir('runs/')
    checkpoint_dir = str(max([int(dir) for dir in lof]))

    # checkpoint_dir = max(p.glob('*'), key =lambda p:p.stat().st_ctime)#os.path.getctime)
    # print(checkpoint_dir)
    checkpoint_file = tf.train.latest_checkpoint(os.path.join('runs/', checkpoint_dir,'checkpoints/'))#FLAGS.checkpoint_dir)
    graph = tf.Graph()

    with graph.as_default():

        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)

        sess = tf.Session(config=session_conf)

        # global_step = tf.train.get_or_create_global_step()

        with sess.as_default():

            # Load the saved meta graph and restore variables
            # saver = tf.train.Saver()
            saver_meta = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver_meta.restore(sess, checkpoint_file)

            # path = './retrain.tfrecord'
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            sample_weight = graph.get_operation_by_name("loss/sample_weights").outputs[0]
            data_path = graph.get_operation_by_name("data_path").outputs[0]
            iterator_init = graph.get_operation_by_name('iterators/dataset_init')
            acc = graph.get_operation_by_name('accuracy/accuracy').outputs[0]
            loss = graph.get_operation_by_name('loss/add').outputs[0]


            """On recupere les poids du graph deja entraine """
            train_op = tf.get_collection('train_op')[0]

            # gradients = graph.get_operation_by_name('optimizer/gradients')
            # saver2 = tf.train.Saver()
            # saver2.restore(sess, checkpoint_file)
            # optimizer = tf.get_collection("Adam")[0]
            # oh = graph.get_operation_by_name('one_hot').outputs[0]
            # shuffle = graph.get_operation_by_name('shuffle').outputs[0]
            # fname = graph.get_operation_by_name('flat_filenames').outputs[0]

            # saver2 = tf.train.Saver(sess, checkpoint_file)
            # saver2.restore()

            gs = tf.train.get_or_create_global_step()
            increment_global_step = tf.assign_add(gs, 1)



            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            #enlever le merged du dessous pour revenir a resultat classique
            # grad_summaries_merged = []

            # for g, v in grads_and_vars:
            #     if g is not None:
            #         grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            #         sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            #         grad_summaries.append(grad_hist_summary)
            #         grad_summaries.append(sparsity_summary)
            grad_summaries_merged = []#tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = checkpoint_dir#'save_dir_for_test'#str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", loss)
            acc_summary = tf.summary.scalar("accuracy", acc)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # sess.run(iterator_init,feed_dict={data_path: path} )

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints, save_relative_paths=True)

            for epoch in range(FLAGS.num_epochs):

                sess.run(iterator_init, feed_dict={data_path:'./retrain.tfrecord'})

                while True:
                    # As long as the iterator is not empty
                    try:

                        feed_dict = {
                            sample_weight: weights,
                            dropout_keep_prob: FLAGS.dropout_keep_prob
                        }

                        _, global_step, summaries, loss_local, accuracy = sess.run(
                            [train_op, gs, train_summary_op, loss, acc],
                            feed_dict)

                        time_str = datetime.datetime.now().isoformat()

                        train_summary_writer.add_summary(summaries, global_step)

                        # current_step = tf.train.global_step(sess, global_step)


                        if global_step % FLAGS.checkpoint_every == 0:
                                path = saver.save(sess, checkpoint_prefix, global_step=global_step)
                                print("Saved model checkpoint to {}\n".format(path))

                        if global_step % 10 == 0:
                            # inp, out, lengths = sess.run([M.sequence, M.lm_preds, M.lengths],
                            #                              feed_dict={M.lr: lr, M.keep_prob: keep_prob})
                            print("************************************************")
                            print("train step:")
                            print("{}: global_step {}, loss {:g}, acc {:g}".format(time_str, global_step, loss_local, accuracy))
                            print("epoch: "+str(epoch))
                            # print(cohenkappa)
                    except tf.errors.OutOfRangeError:
                        # If the iterator is empty stop the while loop
                        break

                sess.run(iterator_init, feed_dict={data_path:'./val.tfrecord'})

                while True:
                    # As long as the iterator is not empty
                    try:

                        feed_dict = {
                            sample_weight: weights,
                            dropout_keep_prob: 1.0
                        }
                         # dev_summary_op,
                        global_step, summaries, loss_local, accuracy, _ = sess.run(
                            [gs, dev_summary_op, loss, acc, increment_global_step],
                            feed_dict)
                        time_str = datetime.datetime.now().isoformat()


                        if global_step % FLAGS.checkpoint_every == 0:
                                path = saver.save(sess, checkpoint_prefix, global_step=global_step)
                                print("Saved model checkpoint to {}\n".format(path))

                        # if global_step % 10 == 0:

                        print("------------------------------------------------")
                        print("validation step:")
                        print("{}: global_step {}, loss {:g}, acc {:g}".format(time_str, global_step, loss_local, accuracy))
                        print("epoch: "+str(epoch))
                            # print(cohenkappa)

                        writer = dev_summary_writer
                        if writer:
                             writer.add_summary(summaries, global_step)
                    except tf.errors.OutOfRangeError:
                        break










def main(argv=None):
    maxlen, vocab_size, num_classes, weights = preprocess()
    retrain(maxlen, vocab_size, num_classes, weights)

if __name__ == '__main__':
    tf.app.run()
