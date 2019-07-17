#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import time
import datetime
# import data_helpers
# import data_and_labels
from text_cnn import TextCNN
import random
# from preparateur import DocPreparateur
import pickle
from os.path import dirname, abspath

# from tflearn.data_utils import VocabularyProcessor

# Parameters
# ==================================================

# Data loading params
# tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
# tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
# tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 301, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")
#
# def tokenizer(x):
#     return x.split(' ')

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

    """batch_size ???"""
    # print(np.tile(weights, ))
    # print("Vocabulary Size: {:d}".format(len(vocab_processor)))
    # print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return maxlen, vocab_size, num_classes, weights

def train(maxlen, vocab_size, num_classes, weights):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)

        global_step = tf.train.get_or_create_global_step()


        cnn = TextCNN(
            global_step=global_step,
            sequence_length=maxlen,
            vocab_size=vocab_size,
            num_classes = num_classes,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        sess = tf.Session(config=session_conf)
        with sess.as_default():



            # with tf.name_scope("optimizer"):
            # sess.run(tf.local_variables_initializer())
            optimizer = tf.train.AdamOptimizer(1e-3)#, name="adam_optimizer")
            # grads_and_vars = optimizer.compute_gradients(cnn.cohenkappa)

            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=cnn.global_step)#, name='train_op')
            tf.add_to_collection('train_op', train_op)



            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            #enlever le merged du dessous pour revenir a resultat classique
            # grad_summaries_merged = []

            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))#'save_dir_for_test'#
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints, save_relative_paths=True)
            saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints, save_relative_paths=True)
            # Initialize all variables
            init = tf.group(tf.global_variables_initializer())#, tf.local_variables_initializer())
            # sess.run(tf.global_variables_initializer())
            ##On fait ça pour resoudre cohen_kappa initialization
            sess.run(init)

            for epoch in range(FLAGS.num_epochs):


                sess.run(cnn.data_init_op, feed_dict={cnn.data_path:'./train.tfrecord'})

                while True:
                    # As long as the iterator is not empty
                    try:

                        feed_dict = {

                            cnn.sample_weight: weights,
                            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                        }

                        _, global_step, summaries, loss, accuracy = sess.run(
                            [train_op, cnn.global_step, train_summary_op, cnn.loss, cnn.accuracy],
                            feed_dict)


                        time_str = datetime.datetime.now().isoformat()

                        train_summary_writer.add_summary(summaries, global_step)

                        # current_step = tf.train.global_step(sess, global_step)


                        if global_step % FLAGS.checkpoint_every == 0:
                                path = saver.save(sess, checkpoint_prefix, global_step=global_step)
                                print("Saved model checkpoint to {}\n".format(path))

                        if global_step % 1 == 0:
                            # inp, out, lengths = sess.run([M.sequence, M.lm_preds, M.lengths],
                            #                              feed_dict={M.lr: lr, M.keep_prob: keep_prob})
                            print("************************************************")
                            print("{}: global_step {}, loss {:g}, acc {:g}".format(time_str, global_step, loss, accuracy))
                            # print(cohenkappa)
                    except tf.errors.OutOfRangeError:
                        # If the iterator is empty stop the while loop
                        break

                sess.run(cnn.data_init_op, feed_dict={cnn.data_path:'./val.tfrecord'})

                while True:
                    # As long as the iterator is not empty
                    try:

                        feed_dict = {
                            cnn.sample_weight: weights,
                            cnn.dropout_keep_prob: 1.0
                        }
                        global_step, summaries, loss, accuracy, _, prob = sess.run(
                            [cnn.global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.increment_global_step, cnn.probas],
                            feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        # print(np.amax(prob, axis = 1))


                        if global_step % FLAGS.checkpoint_every == 0:
                                path = saver.save(sess, checkpoint_prefix, global_step=global_step)
                                print("Saved model checkpoint to {}\n".format(path))

                        # if global_step % 10 == 0:

                        print("------------------------------------------------")
                        print("validation step:")
                        print("{}: global_step {}, loss {:g}, acc {:g}".format(time_str, global_step, loss, accuracy))
                            # print(cohenkappa)

                        writer = dev_summary_writer
                        if writer:
                            writer.add_summary(summaries, global_step)

                    except tf.errors.OutOfRangeError:
                        break

                print("epoch: "+str(epoch))


def main(argv=None):
    maxlen, vocab_size, num_classes, weights = preprocess()
    train(maxlen, vocab_size, num_classes, weights)

if __name__ == '__main__':
    tf.app.run()
