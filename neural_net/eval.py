#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from text_cnn import TextCNN
# from tensorflow.contrib import learn
import csv
import datetime
import os
import pickle
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from os.path import dirname, abspath
import re

# from prepare_dataset import make_dataset

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("eval_data_file", "val.tfrecord", "Data source for the eval data.")

# Eval Parameters
# tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# tf.flags.DEFINE_string("checkpoint_dir", "runs/save_dir_for_test/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
local_dir = abspath(dirname(__file__))
os.chdir(local_dir)

def int_to_class():
    p = '../fonctions_preparation/'

    with open(p+'config', 'r') as config:
        classes = [line.rstrip() for line in config if '#' not in line]
        # lclasses = {classe:numero for elem in classes for classe, numero in elem.split(':')}
        # lclasses = {elem.split(':')[0]:int(elem.split(':')[1]) for elem in classes}
        lclasses = {int(elem.split(':')[1]):elem.split(':')[0] for elem in classes}
    lclasses[0] = '[AUTRE]'
    print(lclasses)
    return lclasses

lclasses = int_to_class()



with open('./metadata.pkl', 'rb') as mt:
        metadata = pickle.load(mt)

num_classes = metadata['num_classes']

print("\nEvaluating...\n")

# Evaluation
# ==================================================
print(os.getcwd())
lof = os.listdir('runs/')

checkpoint_dir = str(max([int(dir) for dir in lof]))
checkpoint_dir = os.path.join('runs/', checkpoint_dir,'checkpoints/')
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)#FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables

        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        path = './eval.tfrecord'
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate

        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        sample_weight = graph.get_operation_by_name("loss/sample_weights").outputs[0]
        data_path = graph.get_operation_by_name("data_path").outputs[0]
        iterator_init = graph.get_operation_by_name('iterators/dataset_init')
        acc = graph.get_operation_by_name('accuracy/accuracy').outputs[0]
        oh = graph.get_operation_by_name('one_hot').outputs[0]
        embedding_shape = graph.get_operation_by_name('embedding/random_uniform/shape').outputs[0]
        probas = graph.get_operation_by_name('output/probas').outputs[0]
        fname = graph.get_tensor_by_name('get_filenames:0')

        sess.run(iterator_init,feed_dict={data_path: path} )

        # Collect the predictions here
        all_predictions = []
        doc_ids = []
        filenames = []
        weights = [1]*num_classes
        assurace_decision = []
        id_documents = []
        idclients = []#np.array(['id_client'])

        txt_pred = []
        #initialisation du dataset que nous voulons tester

        while True:

            #enlever shuffle = False pour melanger
            #mettre True ne marchera pas
            try:
                feed_dict = {
                sample_weight: weights,
                dropout_keep_prob: 1.0}
                print('\ntest\n')
                print(embedding_shape.eval())
                batch_predictions, accuracy, one_hot, es, fn, prob = sess.run([predictions, acc, oh, embedding_shape, fname, probas], feed_dict)
                print('\ntest\n')
                all_predictions = np.concatenate([all_predictions, batch_predictions])

                batch_txt_pred = [lclasses[int(c)] for c in batch_predictions]
                txt_pred = np.concatenate([txt_pred, batch_txt_pred])
                print(batch_txt_pred)
                # for c in batch_predictions:
                #     txt_pred = np.concatenate([txt_pred,lclasses[]])
                # print(batch_predictions.reshape(-1).shape)
                # batch_txt_pred = lclasses[batch_predictions.flatten().astype(int)]

                print("predictions:")
                print(batch_predictions)
                print(es)


                doc_id = [list(o).index(1) for o in list(one_hot)]
                doc_ids = np.concatenate([doc_ids, doc_id])
                fn = [f.decode("utf-8") for f in fn]
                filenames = np.concatenate([filenames, fn])
                print(fn)
                id_document = [re.search('(?<=(])_)([0-9])+', i).group(0) for i in fn]
                idclient = [re.search('(?<=([0-9])_)([0-9])+', i).group(0) for i in fn]
                id_documents = np.concatenate([id_documents, id_document])
                idclients = np.concatenate([idclients, idclient])# = np.concatenate([idclients, idclient])
                assurace_decision = np.concatenate([assurace_decision,np.amax(np.round(prob*100, decimals = 1), axis=1)])
                # filenames = np.concatenate([filenames, filename])
                print("\nreality:")
                print(np.array(doc_id))
                print(np.round(prob*100, decimals = 1))
                print(str(accuracy*100)+"%")
                # print(filename)


            except tf.errors.OutOfRangeError:
                        # If the iterator is empty stop the while loop
                        print("end of dataset")
                        break

        nb_predict = (np.size(all_predictions))


# Save the evaluation to a csv
correct_bool = [0 if a==b else 1 for a,b in zip(all_predictions, doc_ids)]#[np.equal(all_predictions, doc_ids)

predictions_human_readable = np.column_stack((idclients, id_documents,txt_pred, all_predictions, doc_ids, assurace_decision, correct_bool))
print(predictions_human_readable)
dict_results = {}

# for l in predictions_human_readable[1:]:
for l in predictions_human_readable:
    if l[0] in dict_results:
        dict_results[l[0]].append(l)
    else:
        dict_results[l[0]] = [l]
print(dict_results)
for key in dict_results:

    file_name = str(key)+"_predictions.csv"
    out_path = os.path.join("../predictions/", file_name)#"prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    exists = os.path.isfile(out_path)
    if exists:
        with open(out_path, 'a') as f:
            # csv.writer(f).writerow(["id_client", "id_document" ,"filename", "prediction", "reality", "neural_net_confidence", "is_different"])
            csv.writer(f).writerows(dict_results[key])
    else:
        with open(out_path, 'w') as f:
            csv.writer(f).writerow(["id_client", "id_document","txt_prediction", "prediction", "reality", "neural_net_confidence", "is_different"])
            csv.writer(f).writerows(dict_results[key])
#
# with open(out_path, 'w') as f:  # Just use 'w' mode in 3.x
#     w = csv.DictWriter(f, dict_results.keys())
#     w.writeheader()
#     w.writerow(dict_results)


# with open(out_path, 'w') as f:
#     csv.writer(f).writerows(predictions_human_readable)
        # csv.writer(f).writerows(predictions_human_readable)


print(confusion_matrix(doc_ids[1:], all_predictions[1:]))
