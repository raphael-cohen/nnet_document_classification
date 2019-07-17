#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import csv
import json
import re
import shutil
from os import listdir
from os.path import isfile, join

import numpy as np




def loadjson(pathfile, raw_txt, lclasses, list_of_libel):


    with open(pathfile, 'r') as f:
        file = json.load(f)
    reduced_path, json_name = pathfile.rsplit('/',1)
    if 'eval' in reduced_path:
        shutil.move(pathfile, reduced_path+'/archive/'+json_name)
    # """ pour les fichiers incomplets"""
    # idsclient = ['id_client']
    # idsdoc = ['id_doc']
    # libels = ['lib_balise']


    for item in file:

        try:
            libel = re.sub(r' ', '_',item["lib_balise"])
            libel = re.sub(r'\'', '_',libel)
        except:
            #unknown
            libel = '[ERREUR]'
        if len(libel) == 0:
            libel = '[INCONNU]'

        list_of_libel[libel]=0

        name = libel+'_'+item["id_document"]+'_'+item["id_client"]

        try:
            doc_id = lclasses[libel]
        except:
            doc_id = 0

        # property = {'libelle_txt':libel, 'id_doc':item["id_document"], 'id_client':item["id_client"], 'text':item["mots_cles"]}
        property = {'text':item["mots_cles"], 'doc_id':doc_id, 'filename':name}

        if(len(item["mots_cles"]) > 20):

            raw_txt.append(property)



def get_list_path(mypath):

    onlyfiles = [{"path":mypath+f, "name":f} for f in listdir(mypath) if isfile(join(mypath, f)) and ".json" in f]
    return onlyfiles


def retrieve_txt(folder):

    list_of_libel = dict()

    try:
        with open(folder+"/../../liste_des_balises.txt", 'r') as lbal:
            for bal in lbal.readlines():
                list_of_libel[bal.rstrip()]=0
    except:
        print("Construction de la premiere liste de balises")

    listofpath = get_list_path(folder)
    raw_txt = []
    lclasses = class_to_int()

    for path in listofpath:
        loadjson(path["path"], raw_txt, lclasses, list_of_libel)
        print(path['name'])

    with open(folder+'/../../liste_des_balises.txt', 'w') as outfile:
        for w in list_of_libel.keys():
            outfile.write(w + '\n')

    return raw_txt


def class_to_int():

    with open('config', 'r') as config:
        classes = [line.rstrip() for line in config if '#' not in line]
        # lclasses = {classe:numero for elem in classes for classe, numero in elem.split(':')}
        lclasses = {elem.split(':')[0]:int(elem.split(':')[1]) for elem in classes}

    return lclasses
