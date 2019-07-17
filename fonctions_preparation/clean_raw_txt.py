#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import multiprocessing as multi
import re
import string
import xml.etree.ElementTree
from collections import Counter
from multiprocessing import Manager, Process
from os import listdir
from os.path import isfile, join, dirname, abspath
from random import shuffle

import nltk
import numpy as np
import unidecode
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize

from get_raw_txt import retrieve_txt



manager = Manager()
lexique = manager.dict()
mauvais_ocr = manager.list()
clean_txt = manager.list()
## defining the working directory where this file is
local_dir = abspath(dirname(__file__))




def retrieve_all(dicopath, training=True):

    # dico = create_dico(dicopath,'Morphalou-2.0.xml')
    dico = rdico(dicopath)

    #Training
    if training==0:

        """
        dossier du training
        """

        raw_txt = retrieve_txt('../data/train_json/')

        clot_parallele(raw_txt, dico, train = True)

        # print(clean_txt)

        with open('../data/lexique/lexique.txt', 'w') as outfile:
            for w in lexique.keys():
                outfile.write(w + '\n')

        # keys = mauvais_ocr[0].keys()
        # with open('data/txt/mauvais_ocr.csv', 'w') as output_file:
        #     dict_writer = csv.DictWriter(output_file, keys)
        #     dict_writer.writeheader()
        #     dict_writer.writerows(mauvais_ocr)

    #Evaluation
    else:

        try:
            with open(local_dir+'/../data/lexique/lexique.txt', 'r') as lex:
                for w in lex.readlines():
                    lexique[w.rstrip()] = 0

        except Exception as e:
            print(e)
            print("Pas de lexique enregistré, pensez à executer \"to_tfrecord\" avec un json d\'entrainement")

        if training == 1:
            """
            dossier de l'eval
            """
            raw_txt = retrieve_txt('../data/eval_json/')
            clot_parallele(raw_txt, [], train = False)

        #Retrain
        elif training == 2:
            """
            dossier de l'eval
            """
            raw_txt = retrieve_txt('../data/train_json/')
            clot_parallele(raw_txt, [], train = False)

    return clean_txt


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets
    """

    # #on enleve les accents
    string = unidecode.unidecode(string)

    """garder les dates ? les noms propres ? Comment les detecter ?"""

    # string = str(string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"l\'", "", string)
    string = re.sub(r"d\'", "", string)
    string = re.sub(r"m\'", "", string)
    string = re.sub(r"[a-z]{1,1}\'", "", string)

    return string.strip().lower()


def wordtokenize(s):

    s = str(s)
    # delete punctuation
    translate_table = dict((ord(char), None) for char in string.punctuation)
    s = s.translate(translate_table)

    wt = nltk.tokenize.word_tokenize(str(s))

    return wt


# on enleve les mots qui n'apportent pas de sens a la phrase
def stop_words_remove(listofwords):

    filtered_words = [word for word in listofwords if (
        word not in stopwords.words('french')) and len(word) > 1]
    return filtered_words


# construit une liste de mots du dictionnaire
def create_dico(dicopath, name):

    dico = xml.etree.ElementTree.parse(dicopath + name).getroot()
    list_dic = np.array([unidecode.unidecode(str(orth.text))
                         for orth in dico.iter('orthography')])
    with open(dicopath + 'listdico.txt', 'w') as outfile:
        for w in list_dic:
            outfile.write(w + '\n')
    return list_dic

# filtre les mots, ne garde que ceux qui figurent dans le dictionnaire francais
# permet de nettoyer le texte pour ne garder que les mots interessants


def dictionnary_filter(wt, dicolist, lexique, train):

    filtered = []

    for word in wt:

        if word in lexique:
            filtered.append(word)
        ### Faire attention ici, si c'est en train on ne veut pas qu'il prenne
        ### les mots du dictionnaire ...
        elif word in dicolist and train:
            filtered.append(word)

    return filtered


def clean_list_of_texts(raw_txt, dico, lexique, clean_txt, mauvais_ocr, train):

    for rtxt in raw_txt:

        # file = p["path"]
        # name = p["name"]

        # with open(file) as myfile:
        #     txt = myfile.read()
        #     clean_string = clean_str(txt)

        txt = rtxt['text']
        clean_string = clean_str(txt)

        wt = wordtokenize(clean_string)
        wt = stop_words_remove(wt)
        wt = dictionnary_filter(wt, dico, lexique, train)

        """ Au moins 40 mots différents et 100 mots en tout"""
        # if(len(set(wt)) >= 40 and len(wt) > 100):
        if(len(set(wt)) >= 20 and len(wt) > 20):

            # wt = to_most_frequent(wt, 2000)
            # with open(pathtowrite + name, 'w') as f:
            for w in wt:
                lexique[w] = 1

            rtxt['text'] = wt

            clean_txt.append(rtxt)

            # f.write(w + '\n')

            print(rtxt['filename'] + " Cleaned")
            print("size: {0}".format(len(wt)))
        else:
            print(rtxt['filename'] + " Insufficient number of words")
            print("size: {0}".format(len(wt)))
        #     classoffile = re.search('\[(.+?)\]', p["name"]).group(1)
        #     iddoc = re.search('\]_(.+?)_', p["name"]).group(1)
        #     idclient = re.search(iddoc + '_(.+?)\.', p["name"]).group(1)
        #     mauvais_ocr.append(
        #         {'idclient': idclient, 'iddoc': iddoc, 'libel': classoffile})
        #     # print(idclient,' | ', iddoc,' | ', classoffile)


def chunks(n, raw_txt):
    """Splits the list into n chunks"""
    return np.array_split(raw_txt, n)


def clot_parallele(raw_txt, dico, train = False):

    cpus = multi.cpu_count()
    workers = []

    txt_bins = chunks(cpus, raw_txt)

    for cpu in range(cpus):

        worker = multi.Process(name=str(cpu),
                               target=clean_list_of_texts,
                               args=(txt_bins[cpu], dico, lexique, clean_txt, mauvais_ocr, train,))
        worker.start()
        workers.append(worker)

    for worker in workers:
        worker.join()

    print("cleaning ok")

    #raw_txt = clean


def rdico(dicopath):

    with open(dicopath + 'listdico.txt', 'r') as f:
        dico = [word.replace('\n', '') for word in f.readlines()]

    return set(dico)


def to_most_frequent(txt, num):

    fdist = FreqDist(txt).most_common(num)
    return [word[0] for word in fdist]
