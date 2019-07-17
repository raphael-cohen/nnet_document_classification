#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
Tools to take a directory of txt files and convert them to TF records
'''
from collections import Counter, defaultdict

import numpy as np
import tensorflow as tf

PAD = "<PAD>"
START = "<START>"
EOS = "<EOS>"


class Preparateur():
    '''
    Class that converts text inputs to numpy arrays of ids.
    It assigns ids sequentially to the token on the fly.
    '''
    def __init__(self):
        self.vocab = defaultdict(self.next_value)  # map tokens to ids. Automatically gets next id when needed
        self.token_counter = Counter()  # Counts the token frequency
        self.vocab[PAD] = 0
        self.vocab[START] = 1
        self.vocab[EOS] = 2
        self.next = 2  # After 2 comes 3
        # self.tokenizer = tokenizer_fn
        # self.tokentxt = tokentxt
        self.reverse_vocab = {}

    def next_value(self):
        self.next += 1
        return self.next


    def tokenizer(self, x):
        return x.split(' ')

    def sequence_to_tf_example(self, sequence):
        '''
        Gets a sequence (a text like "hello how are you") and returns a a SequenceExample
        :param sequence: Some text
        :return: A A sequence exmaple
        '''
        # Convert the text to a list of ids
        id_list = self.sentence_to_id_list(sequence)
        ex = tf.train.SequenceExample()
        # A non-sequential feature of our example
        sequence_length = len(id_list) + 2  # For start and end
        # Add the context feature, here we just need length
        ex.context.feature["length"].int64_list.value.append(sequence_length)
        # Feature lists for the two sequential features of our example
        # Add the tokens. This is the core sequence.
        # You can add another sequence in the feature_list dictionary, for translation for instance
        fl_tokens = ex.feature_lists.feature_list["tokens"]
        # Prepend with start token
        fl_tokens.feature.add().int64_list.value.append(self.vocab[START])
        for token in id_list:
            # Add those tokens one by one
            fl_tokens.feature.add().int64_list.value.append(token)
        # apend  with end token
        fl_tokens.feature.add().int64_list.value.append(self.vocab[EOS])
        return ex

    def ids_to_string(self, tokens, length=None):
        string = ''.join([self.reverse_vocab[x] for x in tokens[:length]])
        return string

    def convert_token_to_id(self, token):
        '''
        Gets a token, looks it up in the vocabulary. If it doesn't exist in the vocab, it gets added to id with an id
        Then we return the id
        :param token:
        :return: the token id in the vocab
        '''
        self.token_counter[token] += 1
        return self.vocab[token]

    def sentence_to_tokens(self, sent):
        return self.tokenizer(sent)

    def tokens_to_id_list(self, tokens):
        return list(map(self.convert_token_to_id, tokens))

    def sentence_to_id_list(self, sent):
        """ tokens = sent avec le changement de procedure pour clean les txt """
        """ on a directement la liste sous forme de tableau de mots, plus besoin de tokenizer """
        tokens = sent #self.sentence_to_tokens(sent)
        id_list = self.tokens_to_id_list(tokens)
        return id_list

    def sentence_to_numpy_array(self, sent):
        id_list = self.sentence_to_id_list(sent)
        return np.array(id_list)

    def update_reverse_vocab(self):
        self.reverse_vocab = {id_: token for token, id_ in self.vocab.items()}

    def id_list_to_text(self, id_list):
        tokens = ''.join(map(lambda x: self.reverse_vocab[x], id_list))
        return tokens

    @staticmethod
    def parse(ex):
        '''
        Explain to TF how to go from a serialized example back to tensors
        :param ex:
        :return: A dictionary of tensors, in this case {seq: The sequence, length: The length of the sequence}
        '''
        context_features = {
            "length": tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }

        # Parse the example (returns a dictionary of tensors)
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )
        return {"seq": sequence_parsed["tokens"],
                "length": context_parsed["length"]}


class DocPreparateur(Preparateur):
    '''
    An extension of Preppy suited for our task of the table.
    It adds
    1) Storing the book_id in the TFRecord
    2) A map from book_ids to book names so we can explore the results
    '''
    def __init__(self, pad_len):
        super(DocPreparateur, self).__init__()
        self.pad_len = pad_len
        # self.book_map = {}

    def sequence_to_tf_example(self, sequence, doc_id, filename):
        id_list = self.sentence_to_id_list(sequence)
        #On met tous les textes a la même longueur
        id_list = tf.keras.preprocessing.sequence.pad_sequences([id_list], maxlen=self.pad_len, dtype='int64', padding='post', truncating='post')[0]

        ex = tf.train.SequenceExample()
        # A non-sequential feature of our example
        # sequence_length = len(sequence)
        sequence_length = len(id_list)

        """ A changer """

        # A changer dans data_and_labels pour eviter d'avoir a faire ca
        # doc_id = list(doc_id).index(1)
        # print(doc_id)
        # ex.context.feature["length"].int64_list.value.append(sequence_length + 2)  # +2 for start and end
        # ex.context.feature["length"].int64_list.value.append(self.pad_len + 2)
        ex.context.feature["length"].int64_list.value.append(self.pad_len)
        ex.context.feature["doc_id"].int64_list.value.append(doc_id)
        ex.context.feature["filename"].bytes_list.value.append(filename)
        #extend
        # Feature lists for the two sequential features of our example
        fl_tokens = ex.feature_lists.feature_list["tokens"]
        """ a remettre eventuellement """
        # fl_tokens.feature.add().int64_list.value.append(self.vocab[START])
        for token in id_list:
            fl_tokens.feature.add().int64_list.value.append(token)
        # fl_tokens.feature.add().int64_list.value.append(self.vocab[EOS])
        return ex

    @staticmethod
    def parse(ex):
        '''
        Explain to TF how to go from a serialized example back to tensors
        :param ex:
        :return:
        '''
        context_features = {
            "length": tf.FixedLenFeature([], dtype=tf.int64),
            "doc_id": tf.FixedLenFeature([], dtype=tf.int64),
            "filename": tf.FixedLenFeature([], dtype=tf.string, default_value="")
        }
        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }

        # Parse the example (returns a dictionary of tensors)
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )
        return {"seq": sequence_parsed["tokens"], "length": context_parsed["length"],
                "doc_id": context_parsed["doc_id"], "filename": context_parsed["filename"]}
