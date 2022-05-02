from misc_lib import *
import numpy as np
import json
import itertools as itert
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

# from nltk.parse.stanford import StanfordDependencyParser
from stanfordcorenlp import StanfordCoreNLP
from ranking_builder import *
from pos_processes import *
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef

from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
# nltk.download('wordnet')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')

from sklearn import metrics


def stems(term):
    return stemmer.stem(term.lower())
def lemms(term):
    return lemmatizer.lemmatize(term.lower())
def normalize(term):
    return stems(lemms(term.lower()))


class Annotate:
    def __init__(self,score_col='mcc', hp=None, expand_low_entities=False):
        self.hp = hp
        self.posp = hp.posp
        self.file_prefix = self.posp.file_prefix
        self.score_col = score_col
        self.expand_low_entities = expand_low_entities
        self.extracted_entities = None
        self.label_mapping = None

    def load_rule_weights(self):
        self.rule_weights = pd.read_csv('output/'+self.file_prefix+'/models/rule_ranking_merged_on_%s.csv'%(self.score_col))

    def load_annotations(self):
        self.annotations = pd.read_csv('output/'+self.file_prefix+'/stats/tmp3_obj.csv')

    def score_cols_and_mapping(self):
        score_cols = ['need_satisfier_in_o_1',         'need_satisfier_in_o_2',         'need_satisfier_in_o_3',    'need_in_o_8',         'need_in_o_9',   'need_satisfier_in_o_11',         'need_satisfier_in_s_12',         'need_satisfier_in_o_13', 'current_state_in_s_14',         'service_description_in_o_15',         'need_satisfier_in_o_16',         'service_in_o_17',         'need_satisfier_description_in_o_18',         'service_description_in_o_19',         'desired_state_in_o_20',    'need_satisfier_in_o_22',         'service_description_in_o_23',         'service_description_in_o_24',         'need_satisfier_in_o_25',    'required_for_in_s_27',         'required_criteria_in_o_28',         'eligibile_criteria_in_s_29',         'eligibile_for_in_o_30',         'need_satisfier_in_o_31',         'need_satisfier_description_in_o_32',         'service_description_in_o_33',         'client_description_in_p_34',         'program_in_s_35',         'need_satisfier_in_o_36',   'service_description_in_s_41',         'client_description_in_o_42',         'service_description_in_s_43', 'client_description_in_o_43', 'program_in_s_44',         'need_satisfier_in_o_44',  'program_in_s_45', 'service_description_in_o_45',      'need_satisfier_in_s_46',         'client_in_o_47']

        sheets = ['program name','service_description', 'required_criteria','need_satisfier','need_satisfier_description','client demographic','client_description',
                'desired_state (outcome)','need']
        org_sheets = ['program','service_description','required_criteria','need_satisfier','need_satisfier_description', 'client', 'client_description','desired_state','need']
        mapping = {}
        for i, j in zip(sheets, org_sheets):
            mapping.setdefault(i, []).append(j)
        return score_cols,mapping, sheets, org_sheets

    def extract_entities(self):
        annot = self
        res_df = annot.rule_weights.copy()
        df = annot.annotations.copy()
        df['N'] = 1
        score_cols,mapping,_,_ = annot.score_cols_and_mapping()
        savedir = 'output/'+annot.file_prefix+'/models/'

        unique_cat = list(mapping.keys())
        unique_cat.sort()
        colors = {'correct': 'g', 'incorrect': 'r'}
        col = 'mcc'
        # fig,ax = plt.subplots(3,3, figsize=(8,8))
        # fig.suptitle("NER Hypothesis Evaluation")
        # xy = np.resize(unique_cat, ax.shape)
        # [axn.set_ylabel('N',rotation=0) for axn in ax[:,0]]
        # [axn.set_xlabel('Evaluation') for axn in ax[-1]]
        entity_cols = {}
        annot.label_mapping = {}
        for k in unique_cat:
            entity_cols[k] = [[k, k+'_phrase']]
            cat_title = k
            if cat_title == 'client demographic':
                cat_title = 'client characteristic'
            cat_title = cat_title.replace('_', ' ').title()
            annot.label_mapping[k] = cat_title
            descs = mapping[k]
            grp = df#df[df.ranked_cat == k].copy()

            cat_cols = []
            for desc in descs:
                cat_cols.append([(re.sub(r'.*_([spo]_[0-9]+)$', r'\1',c),c) for c in score_cols if desc+'_in' in c])
            cat_cols = dict(flatten(cat_cols))

            cc = res_df[(res_df.rule.isin(cat_cols.keys()))]
            cat = cc.iloc[0]['cat']
            aggr = res_df[(res_df['cat']==cat)&(res_df['rule']=='Aggregate')].iloc[0]
            ignore_score = annot.expand_low_entities and aggr[col]<=0
            if not ignore_score:
                cc = res_df[(res_df['mcc']>0.0)&(res_df.rule.isin(cat_cols.keys()))]
                # ccn = res_df[(res_df['mcc']<0.0)&(res_df.rule.isin(cat_cols.keys()))]

            cs = res_df[(res_df.rule.isin(cat_cols.keys()))]
            for c,mcc in cs[['rule','mcc']].values:
                grp[c] = grp[cat_cols[c]] * mcc

            
            cc['slot'] = cc.rule.apply(lambda r: r.split('_')[0])

            grp[col] = 0

            for slot,rules in cc.groupby('slot'):
                if not ignore_score:
                    correct = grp[grp[rules.rule].gt(0).any(axis=1)].index
                else:
                    correct = grp[grp[cat_cols.values()].gt(0).any(axis=1)].index
                grp.loc[correct,col] = 1
                grp.loc[correct, k] =grp.loc[correct]['token_%s'%(slot)]
                if slot=='p':
                    grp.loc[correct, k+'_phrase'] =grp.loc[correct]['word_%s'%(slot)]
                else:
                    grp.loc[correct, k+'_phrase'] =grp.loc[correct]['parsed_phrase_%s'%(slot)]
                
        df.to_csv('output/'+annot.file_prefix+'/models/extracted_entities.csv', index=False)
        annot.extracted_entities = df.copy()

    def generate_annotation_file(self):
        annot = self
        savedir = 'output/'+annot.file_prefix+'/models'
        df = annot.extracted_entities.copy()
        posp = annot.posp

        indexed_sentences = posp.indexed_sentences
        _,mapping,_,_ = annot.score_cols_and_mapping()
        unique_cat = list(mapping.keys())
        unique_cat.sort()
        res = pd.DataFrame(columns=['idx'])
        for k in unique_cat:
            if k not in df.columns:
                continue
            grp = df[~df[k].isnull()].copy()
            if grp.shape[0]==0:
                continue
            tmp = grp.groupby(['idx',k], as_index = False).agg({k+"_phrase":set})
            res = res.merge(tmp,on=['idx'],how='outer')
        
        annotations_dict = {}
        for _,r in res.iterrows():
            text = indexed_sentences[indexed_sentences.idx==r.idx].iloc[0].indexed_text
            terms = text.split('#DEL#')
            rris = {}
            for k in unique_cat:
                if k not in r.index or r[k] != r[k]:
                    continue
                if k not in rris.keys():
                    rris[k] = []
                for rr in r[k].split(','):
                    rris[k].append(terms.index(rr))

            re_compiled = re.compile(r'(.+)(\-[0-9]+)')
            text = ''
            labels = []
            for i,term in enumerate(terms):
                match = re.search(re_compiled,term)
                hit, hi = match[1],match[2]
                for k,rr in rris.items():
                    if i in rr:
                        tstart = len(text)
                        tend = tstart + len(hit)
                        labels.append([tstart,tend,k,hit,list(r[k+"_phrase"])])
                text += hit + ' '

            if r.idx not in annotations_dict.keys():
                annotations_dict[r.idx] = {'text':text,'label':[]}
            annotations_dict[r.idx]['label'].append(labels)

        annotations = []
        for idx,v in annotations_dict.items():
            labels = flatten(v['label'])
            spans = []
            for label in labels:
                tmp = []
                k = label[2]
                cat_title = self.label_mapping[k]
                for phrase in label[4]:
                    combs = list(itert.permutations(phrase.split(' ')))
                    for comb in combs:
                        regex = ("(%s)"%('(( |\\|\~|\-|\n|\+|&){1,3}(%s)*){1,2}'%('|'.join(STOPWORDS))).join(comb))

                        spans_tmp = [g.span() for g in list(re.finditer(regex, v['text'], re.IGNORECASE))]
                        spans_tmp = mergeIntervals(spans_tmp)
                        tmp.append(spans_tmp)
                tmp = flatten(tmp)
                tmp = mergeIntervals(tmp)
                spans.append([t+[cat_title] for t in tmp])
            spans = dedup(flatten(spans))
            

            annotations.append({"idx":idx,"text":v['text'], "label":spans})
            annot.annotations = annotations

        with open(savedir+'/annotations.json', 'w') as file:
            for ann in annotations:
                json.dump(ann, file)
                _=file.write('\n')

        print("Saved to %s"%(savedir+'/annotations.json'))

        # re_compiled = re.compile(r'(.*)\-([0-9]+)')
        # def assign_term(res,term):
        #     if term[1] != 'NONE':
        #         match = re.search(re_compiled,term[0])
        #         if match:
        #             res[int(match[2])] = match[0]
        #     return res
        # res = [None]*len(triples)*2
        # for _,[s,p,o] in triples:
        #     assign_term(res,s)
        #     # assign_term(res,p)
        #     assign_term(res,o)

