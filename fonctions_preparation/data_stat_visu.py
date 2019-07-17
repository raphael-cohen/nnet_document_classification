# import json
import os
import re
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from nltk.probability import FreqDist


def main():

    dir_path = cwd = os.getcwd()  # os.path.dirname(os.path.realpath(__file__))
    # mypath = '/media/Onedrive/LaFonciereNumerique/contract_classification/data/txt/nettoyes/'
    mypath = 'data/txt/nettoyes/'
    out_path = os.path.join(dir_path, mypath)
    listofpath = get_list_path(mypath)
    ltxt = txt_to_list(listofpath)
    bar_plot_size(ltxt)



def get_list_path(mypath):

    onlyfiles = [{"path": mypath + f, "name": f}
                 for f in listdir(mypath) if isfile(join(mypath, f)) and ".txt" in f]
    return onlyfiles


def txt_to_list(listofpath):

    # ltxt = []
    ltxt = {}

    for i, p in enumerate(listofpath):
        with open(p["path"], 'r') as f:
            # classoffile = re.search('[0-9]{1,3}', p["name"])[0]
            try:
                classoffile = re.search('\[(.+?)\]', p["name"]).group(1)
            except AttributeError:
                # AAA, ZZZ not found in the original string
                classoffile = 'erreur'  # apply your error handling

            lines = [l.replace('\n', '') for l in f.readlines()]
            entry = (classoffile, lines)

            if(classoffile in ltxt.keys()):
                ltxt[classoffile].append(lines)

            else:
                ltxt[classoffile] = [lines]

    return ltxt


def bar_plot_size(ltxt):

    # for x, txt in enumerate(ltxt):
    nbofitem = []
    height = []
    keys = []
    for key in ltxt.keys():

        h = np.mean([len(k) for k in ltxt[key]])
        # height.append(h)
        plt.xticks(rotation='vertical')
        plt.bar(key, h)

        nbofitem.append(len(ltxt[key]))

    print(height)

    plt.figure()
    plt.xticks(rotation='vertical')
    plt.bar(ltxt.keys(), nbofitem)

    plt.show()


def freqword_batch(listofpath, classe, num):

    for path in listofpath:
        # print(path["name"])
        if classe in path["name"]:
            with open(path["path"], 'r') as file:
                txt = [word.replace('\n', '') for word in file.readlines()]
                # txt = file.read().replace('\n', '')
                freqword(txt, path["name"], num)


def freqword(txt, name, num):

    fdist = FreqDist(txt)
    # for word in txt:
    #     fdist.inc( word )
    l = fdist.most_common(num)
    print(name + ': \n', l, '\n')
    print(len(l), '\n')


if __name__ == '__main__':
    main()
