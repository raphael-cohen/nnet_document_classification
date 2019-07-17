#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
# from sklearn.metrics import cohen_kappa_score
from prepare_dataset import make_dataset, prepare_dataset_iterators


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, global_step, sequence_length, vocab_size, num_classes, embedding_size, filter_sizes,
            num_filters, l2_reg_lambda=0.0):

        self.data_path = tf.placeholder(tf.string, name="data_path")

        """
        We load the dataset from tfrecord
        """

        batch_size = 32
        dataset = make_dataset(self.data_path, batch_size=batch_size)

        """
        Creation of an iterator to iterate over the dataset
        """
        with tf.name_scope("iterators"):
            self.iterator = tf.data.Iterator.from_structure(
                dataset.output_types, dataset.output_shapes)
            input = self.iterator.get_next()
            self.data_init_op = self.iterator.make_initializer(
                dataset, name='dataset_init')

        """
        x = the whole text
        y = class of the text (bail, constat d affichage etc....)
        """

        self.input_x = input['seq']
        self.input_y = input['doc_id']

        # with tf.name_scope("fnames"):
        self.filenames = tf.reshape(tf.stack(input['filename'], axis = 0), [-1], name = 'get_filenames')
            # self.filenames = tf.constant(input['filename'], name='get_filenames')

        """
        Monitor the learning step
        """
        self.global_step = global_step
        self.increment_global_step = tf.assign(
            self.global_step, self.global_step + 1, name='increment_global_step')  # To increment during val

        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

        # sequence_length.eval()
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        """
        Embedding layer : transformation of words to vectors of floats
        """

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")

            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

            self.embedded_chars_expanded = tf.expand_dims(
                self.embedded_chars, -1)

        """
        Convolution layer. We create a layer for each filter size
        """

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(
                self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.probas = tf.nn.softmax(self.scores, name="probas")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        for_logit = tf.one_hot(self.input_y, num_classes)
        predict_one_hot = tf.one_hot(self.predictions, num_classes)

        with tf.name_scope("loss"):
            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.sample_weight = tf.placeholder(dtype=tf.float32, shape=[None],
                                                name='sample_weights')

            losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.scores, labels=for_logit)
            weights = tf.gather(self.sample_weight, self.input_y)
            # self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            self.loss = tf.reduce_sum(
                losses * weights) / batch_size + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            # correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
