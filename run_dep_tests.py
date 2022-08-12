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

from stanfordcorenlp import StanfordCoreNLP
from pos_processes import *
from ranking_builder import *
from annotate import *

from tree_parser import *

import xlsxwriter

if False:
    col = 'Description'
    program_name = 'Program Name'
    file_prefix='dep_test'
    if True:
        posp = POSProcesses(col=col, program_name=program_name, dep_parser='enhancedpp')
        posp.load_test_sentences(filename='output/dep_test_sentences.txt', file_prefix=file_prefix)


        posp.clean_sentences()


        posp.build_prolog_files()
        posp.run_tests()
        posp.prolog_to_csv()
        _ = os.makedirs('output/%s/models'%(posp.file_prefix)) if not os.path.exists('output/%s/models'%(posp.file_prefix)) else None        
        filehandler = open('output/%s/models/posp_processes.pkl'%(posp.file_prefix), 'wb') 
        pickle.dump(posp, filehandler)
    else:
        filehandler = open('output/%s/models/posp_processes.pkl'%(file_prefix), 'rb') 
        posp = pickle.load(filehandler)

    tp = TreeParser(posp=posp, ext='png', parser_detail=posp.dep_parser)
    tp.draw_sentences(view=False)


    # hp = HypothesisBuilder(posp=posp)
    # hp.gen_phrase_pos_rank()
    # hp.from_mods()

    # annot = Annotate(hp=hp,expand_low_entities=True)
    # annot.load_rule_weights()
    # annot.load_annotations()
    # annot.extract_entities()
    # annot.generate_annotation_file()

