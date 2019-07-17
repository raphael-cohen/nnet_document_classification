#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import subprocess
import os
from os.path import dirname, abspath

print(sys.version)
print(sys.path)

local_dir = abspath(dirname(__file__))
print(local_dir)

try:
    uarg = str(sys.argv[1])
except:
    uarg = 'autre'

d = {'train':0, 'eval':1, 'retrain': 2}

try:
    clean = int(sys.argv[2])
    clean = 0
except:
    clean = 1


if clean == 1 and uarg != 'autre':
    subprocess.call(["python3",local_dir+"/fonctions_preparation/to_tfrecord.py",str(d[uarg])])#+" "+str(uarg))


if uarg == 'train':

    subprocess.call(["python3",local_dir+"/neural_net/train.py"])

elif uarg == 'eval':

    subprocess.call(["python3",local_dir+"/neural_net/eval.py"])


elif uarg == 'retrain':
    subprocess.call(["python3",local_dir+"/neural_net/retrain.py"])

else:
    # print("Merci de rentrer un argument. 0 pour train, 1 pour evaluation, 2 pour le retrain et 3 pour train + evaluation")
    print("Merci de rentrer un argument. train pour l'entrainement, eval pour evaluation et retrain pour le re-entrainement\nDeuxieme argument: ne rien mettre pour préparer les données et mettre 1 pour ne pas préparer les données")
