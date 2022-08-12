from misc_lib import *
import numpy as np
import tqdm
import copy
import pandas as pd
import itertools
import nltk
import spacy
import pickle
import json
import unidecode
from matplotlib import pyplot as plt
import seaborn as sns
import os
# os.chdir("/Users/chaerinsong/Desktop/CSSE/scroll/prolog")
# print(os.getcwd())

# from nltk.parse.stanford import StanfordDependencyParser
from stanfordcorenlp import StanfordCoreNLP
from pos_processes import *
from ranking_builder import *
from annotate import *

import xlsxwriter

if True:
    col = 'Description'
    program_name = 'Program Name'
    if True:
        posp = POSProcesses(col=col, program_name=program_name)
        posp.load_test_sentences(filename='output/alberta_full.csv', file_prefix='alberta_output')
        # posp.clean_sentences()  # Duplicate in build_prolog_files()
        posp.build_prolog_files()
        posp.run_tests()
        posp.prolog_to_csv()
        _ = os.makedirs('output/%s/models'%(posp.file_prefix)) if not os.path.exists('output/%s/models'%(posp.file_prefix)) else None
        filehandler = open('output/%s/models/posp_processes.pkl'%(posp.file_prefix), 'wb')
        pickle.dump(posp, filehandler)
    else:
        file_prefix='alberta_output'
        filehandler = open('output/%s/models/posp_processes.pkl'%(file_prefix), 'rb')
        posp = pickle.load(filehandler)

    # tp = TreeParser(posp=posp)
    # tp.draw_sentences()

    hp = HypothesisBuilder(posp=posp)
    hp.gen_phrase_pos_rank()
    hp.from_mods()

    annot = Annotate(hp=hp,expand_low_entities=True)
    annot.load_rule_weights()
    annot.load_annotations()
    annot.extract_entities()
    annot.generate_annotation_file()


