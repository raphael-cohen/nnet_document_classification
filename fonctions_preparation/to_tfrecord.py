#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# nettoyer les fichiers text
# sauvegarder les metadatas pour une utilisation lors de l'entrainement ou de l'Evaluation
import pickle
import random
import sys
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import os # a enlever
from os.path import dirname, abspath

## defining the working directory where this file is
local_dir = abspath(dirname(__file__))
os.chdir(local_dir)

# pour ecrire les tfrecords qui seront utilisés pour l'apprentissage du model
import tensorflow as tf
import clean_raw_txt
from preparateur import DocPreparateur







DICOPATH = '../dictionnaire_fr/'


def main():

    try:

        user_arg = int(sys.argv[1])
        if user_arg >= 0 and user_arg <=3:

            if user_arg == 3:
                to_tfrecord(train=0)
                to_tfrecord(train=1)

            else:
                to_tfrecord(user_arg)

        else:
            print("Merci de rentrer un argument. 0 pour train, 1 pour evaluation, 2 pour le retrain et 3 pour train + evaluation")


    except Exception as e:
        print(e)
        print("Merci de rentrer un argument. 0 pour train, 1 pour evaluation, 2 pour le retrain et 3 pour train + evaluation")



def to_tfrecord(train=0):

    try:

        """ mettre l'assignment de maxlen et num_classes dans le if(train)"""

        examples = clean_raw_txt.retrieve_all(DICOPATH, train)
        # maxlen = max([len(x['text'].split(' ')) for x in examples])
        maxlen = max([len(elem['text']) for elem in examples])
        list_of_classes = [ex['doc_id'] for ex in examples]
        unique_list_of_classes = list(set(list_of_classes))

        num_classes = max(list_of_classes) + 1

        nb_elem = len(list_of_classes)
        # counter = Counter(list_of_classes)
        # weights = [0] * num_classes
        # for k in counter.elements():
        #     weights[k] = 1 - (counter[k] / nb_elem)  # 1-(counter[k]/nb_elem)
        weights = compute_class_weight(class_weight='balanced', classes=unique_list_of_classes, y=list_of_classes)

    except Exception as e:
        print(e)
        """ Modifier cette exception """
        print("Dossier vide ou chemin incorrect")

    if(train==0):

        DP = DocPreparateur(pad_len=maxlen)

        random.shuffle(examples)
        dev_sample_index = -1 * int(0.1 * float(len(examples)))
        train, val = examples[:dev_sample_index], examples[dev_sample_index:]

        for (data, path) in [(val, '../neural_net/val.tfrecord'), (train, '../neural_net/train.tfrecord')]:
            with open(path, 'w') as f:
                writer = tf.python_io.TFRecordWriter(f.name)
            for example in data:
                record = DP.sequence_to_tf_example(
                    sequence=example['text'], doc_id=example['doc_id'], filename=example['filename'].encode())
                writer.write(record.SerializeToString())

        DP.update_reverse_vocab()
        vocab_size = len(DP.vocab.items())
        # On sauvegarde le preparateur
        pickle.dump(DP, open('../neural_net/preparateur.pkl', 'wb'))
        metadata = {'maxlen': maxlen, 'num_classes': num_classes,
                    'vocab_size': vocab_size, 'weights': weights}
        print(metadata)
        pickle.dump(metadata, open('../neural_net/metadata.pkl', 'wb'))

    elif(train == 1):

        try:
            with open('../neural_net/preparateur.pkl', 'rb') as dp:
                DP = pickle.load(dp)

            for (data, path) in [(examples, '../neural_net/eval.tfrecord')]:
                with open(path, 'w') as f:
                    writer = tf.python_io.TFRecordWriter(f.name)
                for example in data:
                    record = DP.sequence_to_tf_example(
                        sequence=example['text'], doc_id=example['doc_id'], filename=example['filename'].encode())
                    writer.write(record.SerializeToString())

        except Exception as e:
            print(e, '\n')
            print("vous devez avoir un fichier preparateur.pkl (faire un entrainement) avant d'evaluer des donnees")

    elif(train == 2):

        try:
            with open('../neural_net/preparateur.pkl', 'rb') as dp:
                DP = pickle.load(dp)

            for (data, path) in [(examples, '../neural_net/retrain.tfrecord')]:
                with open(path, 'w') as f:
                    writer = tf.python_io.TFRecordWriter(f.name)
                for example in data:
                    record = DP.sequence_to_tf_example(
                        sequence=example['text'], doc_id=example['doc_id'], filename=example['filename'].encode())
                    writer.write(record.SerializeToString())

        except Exception as e:
            print(e, '\n')
            print("vous devez avoir un fichier preparateur.pkl (faire un entrainement) avant de reprendre l'entrainement")



if __name__ == '__main__':
    main()
