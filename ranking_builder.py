from misc_lib import *
import re, os, tqdm, glob
from datetime import date
import pandas as pd, numpy as np
import pickle, json, yaml, unidecode
from nltk import Tree,ParentedTree
from nltk.tokenize import sent_tokenize
from nltk.parse.stanford import StanfordDependencyParser
from graphviz import Source
from stanfordcorenlp import StanfordCoreNLP
from matplotlib import pyplot as plt
import seaborn as sns
import openpyxl
import xlsxwriter
class Entities:
    def __init__(self):
        self.entities = None
        self.nlp = StanfordCoreNLP('http://localhost',9000)

    def load_states_and_satisfiers(self):
        self.entities = pd.read_csv('data/client_states_and_satisfier.csv')

    def extract_tokens(self):
        # extract bracketed qualifier
        pattern = re.compile(r'^\(([^\)]+)\)')
        entities = self.entities
        entities['qualifier'] = entities['text'].apply(lambda text: (re.search(pattern,text) or [None])[0])
        pattern = re.compile(r'^\([^\)]+\)(.+)')
        entities['text'] = entities['text'].apply(lambda text: (re.search(pattern,text) or [None,text])[1].strip())
        # tqdm.tqdm.pandas()
        tokens = []
        for category,grp in self.entities.groupby(['category']):
            tmp = grp['text'].apply(lambda text: self.parse_nlp(text))
            df = pd.DataFrame(flatten(tmp),columns=['token','pos'])
            df['N'] = 1
            df = df.groupby(['token','pos']).count()['N'].reset_index(drop=False)
            df['category'] = category
            _=tokens.append(df)
        tokens = pd.concat(tokens)
        self.tokens = tokens[tokens.pos.str.startswith(('VB','NN','JJ'))]


    def parse_nlp(self,text):
        output = json.loads(
            unidecode.unidecode(
                self.nlp.annotate(unidecode.unidecode(text), properties= {'annotators':'dcoref','outputFormat':'json','ner.useSUTime':'false'})
            )
        )
        return [(t['word'],t['pos']) for t in output['sentences'][0]['tokens']]
        # return output

class TaggedEntities:
    def __init__(self):
        self.entities = None
        self.nlp = StanfordCoreNLP('http://localhost',9000)

    def load_states_and_satisfiers(self):
        self.entities = pd.read_excel('data/client_states_and_satisfier_tagged.xlsx').fillna('')

    def extract_phrases(self):
        phrases = []
        triples = []
        for ix,(qs,ks,pqs) in self.entities[['qualifiers','keywords', 'postqualifiers']].iterrows():
            for q,k,pq in combos_prod([qs.split(','), ks.split(','),pqs.split(',')]):
                q,k,pq = q.strip(),k.strip(),pq.strip()
                phrases.append([ix,k])
                # phrases.append([ix,' '.join([q,k])])
                phrases.append([ix,' '.join([q,k,pq])])
                triples.append([ix,q,k,pq])

        phrases = [[i,' '.join(t.split())] for i,t in phrases]
        phrases = pd.DataFrame(dedup(phrases),columns=['index','phrase'])
        triples = pd.DataFrame(dedup(triples),columns=['index','pre','keyword','post'])

        phrases = self.entities[['category', 'type', 'level-1', 'level-2']].reset_index(drop=False). \
            merge(phrases, on='index', how='left')
        triples = self.entities[['category', 'type', 'level-1', 'level-2']].reset_index(drop=False). \
            merge(triples, on='index', how='left')


        self.phrases = phrases
        self.triples = triples

    def tag_text(self,res_df):
        te = self
        found_phrases = []
        for ix,grp in tqdm.tqdm(res_df.groupby(['idx','phrase_id'])[['word']]):
            found_phrases.append(list(ix)+[' '.join([str(xx).strip() for xx in grp['word'].values])])
        found_phrases = pd.DataFrame(found_phrases, columns=['idx','phrase_id','parsed_phrase'])
        tmp0 = res_df.merge(found_phrases, on=['idx','phrase_id'])

        tmp_keywords = te.triples.drop(columns=['post']).merge(tmp0,left_on=['keyword'],right_on=['word'])
        tmp_pres = te.triples.drop(columns=['post','keyword']).merge(tmp0,left_on=['pre'],right_on=['word'])
        tmp_posts = te.triples.drop(columns=['pre','keyword']).merge(tmp0,left_on=['post'],right_on=['word'])

        tmp2 = tmp_keywords.merge(tmp_posts,on=['idx','phrase_id','index','category','type','level-1','level-2','parsed_phrase'], suffixes=['_mod','_term'])
        tmp2_ = tmp_keywords.merge(tmp_pres,on=['idx','phrase_id','index','category','type','level-1','level-2','parsed_phrase'], suffixes=['_mod','_term'])
        tmp2_ = tmp2_[tmp2_['pre_mod'] != tmp2_['pre_term']]
        tmp2_['post'] = tmp2_['keyword']
        tmp2_['keyword'] = tmp2_['pre_term']
        tmp2_ = tmp2_.drop(columns=['pre_term'])
        tmp2 = tmp2.append(tmp2_).reset_index(drop=True)




    def parse_nlp(self,text):
        output = json.loads(
            unidecode.unidecode(
                self.nlp.annotate(unidecode.unidecode(text), properties= {'annotators':'dcoref','outputFormat':'json','ner.useSUTime':'false'})
            )
        )
        return [(t['word'],t['pos']) for t in output['sentences'][0]['tokens']]
        # return output

class HypothesisBuilder:
    def __init__(self,posp, run_label=date.today().strftime("%d_%b_%Y")):
        self.posp = posp
        self.run_label = run_label
        self.load_parser_rankings()
    def load_parser_rankings(self):
        self.parser_ranking = yaml.safe_load(open('prolog/parsing_ranking.yml','r'))


    def build_records(self):
        print('.',end='',flush=True)
        self.load_entities()
        print('.',end='',flush=True)
        self.rank_corefs_hypothesis()
        print('.',end='',flush=True)
        self.rank_phrases_hypothesis()
        print('.',end='',flush=True)
        self.rank_spo_pos_hypothesis()
        print('.',end='',flush=True)
        self.rank_spo_token_hypothesis()
        print('.',end='',flush=True)
        self.rank_spo_qualifier_token_hypothesis()
        print('.',end='',flush=True)
        self.rank_spo_qualifier_hypothesis()
        print('.',end='',flush=True)
        self.rank_spo_qualifier_roles_types_hypothesis()
        print('.',end='',flush=True)
        self.rank_spo_qualifier_ngram_hypothesis()
        print('.',end='',flush=True)
        self.rank_spo_pos_qualifier_hypothesis()
        print('.',end='',flush=True)
        self.build_highest_spo_qualifier_phrases_hypothesis()
        print('.',end='',flush=True)
        self.spo_qualfiers_csv_to_prolog()
        print('.',end='',flush=True)
        self.spo_qualifier_pos_by_state_satsfier()
        print()

    def rank_predicates(self):
        filename='models/sent_pos_full_Alberta.pkl'
        filehandler = open(filename, 'rb') 
        sent_pos = pickle.load(filehandler)
        roots_scores = pd.DataFrame(flatten([[tt.lower() for tt in t] for t in sent_pos.roots]), columns=['token']).value_counts()
        self.roots_scores = roots_scores/roots_scores.max()


    def res_file_to_df(self, filename, remove_text=True):
        with open(filename, 'r') as temp_f:
            # get No of columns in each line
            col_count = [ len(l.split("\t")) for l in temp_f.readlines() ]

        ### Generate column names  (names will be 0, 1, 2, ..., maximum columns - 2)
        ### Assuming the first colum will be a rankign of resutls: see parsing_ranking.yml
        column_names = ['idx','ranking_ids']+[i for i in range(0, max(col_count)-2)]

        ### Read csv
        df = pd.read_csv(filename, header=None, delimiter="\t", names=column_names)
        if remove_text:
            df=df[df['ranking_ids']!='TEXT'].reset_index(drop=True)

        # get parse ranking for each record
        df['parse_ranking'] = df['ranking_ids'].apply(self.calculate_parse_ranking)
        # remove duplicates, keeping the highest ranked record
        sort_cols = df.columns.tolist()
        sort_cols = [s for s in df.columns if s not in ['parse_ranking','ranking_ids']]
        df = df.loc[df.groupby(sort_cols,dropna=False)['parse_ranking'].idxmax()].sort_index().reset_index(drop=True)
        
        # normalize ranking
        df['parse_ranking'] = df['parse_ranking']/df['parse_ranking'].max()

        return df

    def calculate_parse_ranking(self,ranking_ids):
        ranking_ids = ranking_ids.replace('|',',')
        ids = [float(r) for r in ranking_ids.split(',')]
        return np.sum([self.parser_ranking[i]*w for w,i in zip(np.arange(1,0, -1/len(ids)), ids)])

    def apply_parse_ranking(self,df, group_by_idx=True, cols=['score'], rank_cols=['parse_ranking']):
        df = df.copy()
        if len(cols)>1 and len(rank_cols)>1 and len(rank_cols) != len(cols):
            raise ValueError("col count %s not compatible with rank_col count %s"%(len(cols), len(rank_cols)))
        if len(rank_cols) == 1:
            rank_cols = rank_cols*len(cols)
        for col,rcol in zip(cols,rank_cols):
            df[col] = df[col]*df[rcol]
            if group_by_idx and 'idx' in df.columns:
                for idx,grp in df.groupby('idx'):
                    df.loc[grp.index,col] = grp[col]/grp[col].max()
            else:
                df[col] = df[col]/df[col].max()
        return df


    ########################################################################
    # load files
    ########################################################################
    

    def load_entities(self):
        hp = self
        hp.et = Entities()
        hp.et.load_states_and_satisfiers()
        hp.et.extract_tokens()

    def rank_corefs_hypothesis(self):
        hp  = self
        stats_directory = "output/%s/stats/"%hp.posp.file_prefix
        _ = os.makedirs(stats_directory) if not os.path.exists(stats_directory) else None
            

        # get corefs
        filename = "output/%s/res/coref_results_format.txt"%(hp.posp.file_prefix)
        ranked_corefs = hp.res_file_to_df(filename).rename(columns={0:'ref',1:'coref'}).reset_index(drop=True)

        ranked_corefs = pd.DataFrame(ranked_corefs.apply(lambda row: hp.split_token_pos(row['ref']),axis=1).tolist(), columns=['ref','ref_pos']). \
                join(pd.DataFrame(ranked_corefs.apply(lambda row: hp.split_token_pos(row['coref']),axis=1).tolist(), columns=['coref','coref_pos'])). \
                join(pd.DataFrame(ranked_corefs[['idx','parse_ranking']]))

        ranked_corefs['coref_score(i,t,cr)'] = ranked_corefs['parse_ranking']
        ranked_corefs.to_csv(stats_directory+"corefs_ranked.csv",index=False)
        self.ranked_corefs = ranked_corefs

    def rank_conj_hypothesis(self):
        hp  = self
        stats_directory = "output/%s/stats/"%hp.posp.file_prefix
        _ = os.makedirs(stats_directory) if not os.path.exists(stats_directory) else None
            

        # get corefs
        filename = "output/%s/res/conj_results_format.txt"%(hp.posp.file_prefix)
        ranked_conjs = hp.res_file_to_df(filename).rename(columns={0:'linked',1:'ref'}).reset_index(drop=True)

        ranked_conjs = pd.DataFrame(ranked_conjs.apply(lambda row: hp.split_token_pos(row['ref']),axis=1).tolist(), columns=['ref','ref_pos']). \
                join(pd.DataFrame(ranked_conjs.apply(lambda row: hp.split_token_pos(row['linked']),axis=1).tolist(), columns=['linked','linked_pos'])). \
                join(pd.DataFrame(ranked_conjs[['idx','parse_ranking']]))

        ranked_conjs['linked_score(i,t,cr)'] = ranked_conjs['parse_ranking']
        ranked_conjs.to_csv(stats_directory+"conjs_ranked.csv",index=False)
        self.ranked_conjs = ranked_conjs

    def extract_token(self,val):
        re_pattern = re.compile("(.*),[^,]+")
        match = re.search(re_pattern, val)
        if match is None:
            res = val
        else:
            res = match[1]
        return res
    def split_token_pos(self,val):
        re_pattern = re.compile("(.*),([^,]+)")
        match = re.search(re_pattern, val)
        if match is None:
            res = [val,'']
        else:
            res = [match[1],match[2]]
        return res

    def rank_phrases_hypothesis(self):
        stats_directory = "output/%s/stats/"%self.posp.file_prefix
        _ = os.makedirs(stats_directory) if not os.path.exists(stats_directory) else None
            

        # get phrases
        filename = "output/%s/res/phrase_results_format.txt"%(self.posp.file_prefix)
        df = self.res_file_to_df(filename).rename(columns={0:'phrase'}).reset_index(drop=True)
        ranked_phrases = self.rank_by_length(series=df['phrase'], df=df)
        ranked_phrases = self.apply_parse_ranking(df=ranked_phrases)
        ranked_phrases['label'] = ranked_phrases.apply(lambda row: ' '.join([v.split('-')[0] for v in row['text'].split(',')]), axis=1)
        ranked_phrases.to_csv(stats_directory+"phrases_ranked.csv",index=False)
        self.ranked_phrases = ranked_phrases

    def rank_spo_pos_hypothesis(self):
        stats_directory = "output/%s/stats/"%self.posp.file_prefix
        _ = os.makedirs(stats_directory) if not os.path.exists(stats_directory) else None


        # load file
        filename = "output/%s/res/spo_results_format.txt"%(self.posp.file_prefix)
        df = self.res_file_to_df(filename).rename(columns={0:'s',1:'p',2:'o'}).reset_index(drop=True)
        
        # save pos_stats
        re_pattern = re.compile("(.*),([^,]+)")
        spo_df = pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['s'])[0],axis=1).tolist(), columns=['s_token','spos']). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['p'])[0],axis=1).tolist(), columns=['p_token','ppos'])). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['o'])[0],axis=1).tolist(), columns=['o_token','opos'])). \
                join(pd.DataFrame(df[['idx','parse_ranking']]))

        # save pos-combination stats
        pos_cols = ['spos','ppos','opos']
        counts_spo = spo_df.groupby(pos_cols)[['s_token']].count().rename(columns={'s_token':'N'}). \
                     join(spo_df.groupby(pos_cols)[['parse_ranking']].mean())
        counts_spo['score'] = counts_spo['N']/counts_spo['N'].sum()
        counts_spo = self.apply_parse_ranking(df=counts_spo)


        counts_spo.to_csv(stats_directory+"spo_pos_combinations_ranked.csv",index=True)
        spo_df.merge(counts_spo, on=pos_cols).to_csv(stats_directory+"spo_ranked_by_pos_combination.csv",index=False)
  

    def rank_spo_token_hypothesis(self):
        stats_directory = "output/%s/stats/"%self.posp.file_prefix
        _ = os.makedirs(stats_directory) if not os.path.exists(stats_directory) else None


        # load file
        filename = "output/%s/res/spo_results_format.txt"%(self.posp.file_prefix)
        df = self.res_file_to_df(filename).rename(columns={0:'s',1:'p',2:'o'}).reset_index(drop=True)
        
        # save pos_stats
        re_pattern = re.compile("(.*),([^,]+)")
        spo_df = pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['s'])[0],axis=1).tolist(), columns=['s_token','spos']). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['p'])[0],axis=1).tolist(), columns=['p_token','ppos'])). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['o'])[0],axis=1).tolist(), columns=['o_token','opos'])). \
                join(pd.DataFrame(df[['idx','parse_ranking']]))

        # save ranked token stats
        # spo_df = spo_df.merge(counts_spo, on=['spos','ppos','opos'])
        re_pattern = re.compile("(.*)-[0-9]+$")
        spo_df['sword'] = spo_df['s_token'].apply(lambda row: re.match(re_pattern,row)[1])
        spo_df['pword'] = spo_df['p_token'].apply(lambda row: re.match(re_pattern,row)[1])
        spo_df['oword'] = spo_df['o_token'].apply(lambda row: re.match(re_pattern,row)[1])
        spo_df_stats = spo_df.groupby(['sword'])[['s_token']].count().rename(columns={'s_token':'s_token_N'}). \
            join(spo_df.groupby(['sword'])[['parse_ranking']].mean()).rename(columns={'parse_ranking':'s_parse_ranking'}). \
            merge(spo_df.groupby(['pword'])[['s_token']].count().rename(columns={'s_token':'p_token_N'}), left_index=True,right_index=True,how='outer'). \
            join(spo_df.groupby(['pword'])[['parse_ranking']].mean()).rename(columns={'parse_ranking':'p_parse_ranking'}). \
            merge(spo_df.groupby(['oword'])[['s_token']].count().rename(columns={'s_token':'o_token_N'}), left_index=True,right_index=True,how='outer'). \
            join(spo_df.groupby(['oword'])[['parse_ranking']].mean()).rename(columns={'parse_ranking':'o_parse_ranking'}). \
            fillna(0.0)

        cols = ['s','p','o']
        spo_df_stats['sum_N'] = spo_df_stats[[c+'_token_N' for c in cols]].apply(sum,axis=1)
        for c in cols:
            spo_df_stats[c+'_token_score'] = spo_df_stats[c+'_token_N'] / spo_df_stats['sum_N']
        spo_df_stats = self.apply_parse_ranking(
            df=spo_df_stats, 
            cols=['s_token_score','p_token_score','o_token_score'], 
            rank_cols=['s_parse_ranking','p_parse_ranking','o_parse_ranking'])

        spo_df_stats.index.name="token"
        spo_df_stats.to_csv(stats_directory+"spo_token_ranked.csv",index=True)


    def rank_spo_pos_qualifier_hypothesis(self):
        stats_directory = "output/%s/stats/"%self.posp.file_prefix
        _ = os.makedirs(stats_directory) if not os.path.exists(stats_directory) else None

        # load data
        filename = "output/%s/res/spo_qualifier_results_format.txt"%(self.posp.file_prefix)
        df = self.res_file_to_df(filename).rename(columns={0:'s',1:'p',2:'o',3:'pq',4:'oq'}).reset_index(drop=True)
        # save pos_stats
        re_pattern = re.compile("(.*),([^,]+)")
        spo_df = pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['s'])[0],axis=1).tolist(), columns=['s_token','spos']). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['p'])[0],axis=1).tolist(), columns=['p_token','ppos'])). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['o'])[0],axis=1).tolist(), columns=['o_token','opos'])). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['pq'])[0],axis=1).tolist(), columns=['pq_token','pqpos'])). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['oq'])[0],axis=1).tolist(), columns=['oq_token','oqpos'])). \
                join(pd.DataFrame(df[['idx','parse_ranking']]))


        # save pos-combination stats
        pos_cols = ['spos','ppos','opos','pqpos','oqpos']
        counts_spo = spo_df.groupby(pos_cols)[['s_token']].count().rename(columns={'s_token':'N'}).  \
                     join(spo_df.groupby(pos_cols)[['parse_ranking']].mean())

        counts_spo['score'] = counts_spo['N']/counts_spo['N'].sum()
        counts_spo = self.apply_parse_ranking(df=counts_spo)


        counts_spo.to_csv(stats_directory+"spo_qualifier_pos_combinations_ranked.csv",index=True)
        spo_df.merge(counts_spo, on=pos_cols,how='left').to_csv(stats_directory+"spo_qualifier_ranked_by_pos_combination.csv",index=False)


    def rank_spo_qualifier_token_hypothesis(self):
        stats_directory = "output/%s/stats/"%self.posp.file_prefix
        _ = os.makedirs(stats_directory) if not os.path.exists(stats_directory) else None


        # load file
        filename = "output/%s/res/spo_qualifier_results_format.txt"%(self.posp.file_prefix)
        df = self.res_file_to_df(filename).rename(columns={0:'s',1:'p',2:'o', 3:'pq', 4:'oq'}).reset_index(drop=True)
        
        # save pos_stats
        re_pattern = re.compile("(.*),([^,]+)")
        spo_df = pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['s'])[0],axis=1).tolist(), columns=['s_token','spos']). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['p'])[0],axis=1).tolist(), columns=['p_token','ppos'])). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['o'])[0],axis=1).tolist(), columns=['o_token','opos'])). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['pq'])[0],axis=1).tolist(), columns=['pq_token','pqpos'])). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['oq'])[0],axis=1).tolist(), columns=['oq_token','oqpos'])). \
                join(pd.DataFrame(df[['idx','parse_ranking']]))

        # save ranked token stats
        re_pattern = re.compile("(.*)-[0-9]+$")
        spo_df['sword'] = spo_df['s_token'].apply(lambda row: re.match(re_pattern,row)[1])
        spo_df['pword'] = spo_df['p_token'].apply(lambda row: re.match(re_pattern,row)[1])
        spo_df['oword'] = spo_df['o_token'].apply(lambda row: re.match(re_pattern,row)[1])
        spo_df['pqword'] = spo_df['pq_token'].apply(lambda row: re.match(re_pattern,row)[1])
        spo_df['oqword'] = spo_df['oq_token'].apply(lambda row: re.match(re_pattern,row)[1])
        spo_df_stats = spo_df.groupby(['sword'])[['s_token']].count().rename(columns={'s_token':'s_token_N'}). \
            join(spo_df.groupby(['sword'])[['parse_ranking']].mean()).rename(columns={'parse_ranking':'s_parse_ranking'}). \
            merge(spo_df.groupby(['pword'])[['s_token']].count().rename(columns={'s_token':'p_token_N'}), left_index=True,right_index=True,how='outer'). \
            join(spo_df.groupby(['pword'])[['parse_ranking']].mean()).rename(columns={'parse_ranking':'p_parse_ranking'}). \
            merge(spo_df.groupby(['oword'])[['s_token']].count().rename(columns={'s_token':'o_token_N'}), left_index=True,right_index=True,how='outer'). \
            join(spo_df.groupby(['oword'])[['parse_ranking']].mean()).rename(columns={'parse_ranking':'o_parse_ranking'}). \
            merge(spo_df.groupby(['pqword'])[['s_token']].count().rename(columns={'s_token':'pq_token_N'}), left_index=True,right_index=True,how='outer'). \
            join(spo_df.groupby(['pqword'])[['parse_ranking']].mean()).rename(columns={'parse_ranking':'pq_parse_ranking'}). \
            merge(spo_df.groupby(['oqword'])[['s_token']].count().rename(columns={'s_token':'oq_token_N'}), left_index=True,right_index=True,how='outer'). \
            join(spo_df.groupby(['oqword'])[['parse_ranking']].mean()).rename(columns={'parse_ranking':'oq_parse_ranking'}). \
            fillna(0.0)

        cols = ['s','p','o','pq','oq']
        spo_df_stats['sum_N'] = spo_df_stats[[c+'_token_N' for c in cols]].apply(sum,axis=1)
        for c in cols:
            spo_df_stats[c+'_token_score'] = spo_df_stats[c+'_token_N'] / spo_df_stats['sum_N']
        spo_df_stats = self.apply_parse_ranking(
            df=spo_df_stats, 
            cols=[c+'_token_score' for c in cols], 
            rank_cols=[c+'_parse_ranking' for c in cols])

        spo_df_stats.index.name="token"
        spo_df_stats.to_csv(stats_directory+"spo_qualifier_token_ranked.csv",index=True)


    def rank_spo_qualifier_hypothesis(self):
        stats_directory = "output/%s/stats/"%self.posp.file_prefix
        _ = os.makedirs(stats_directory) if not os.path.exists(stats_directory) else None
        
        # load data
        filename = "output/%s/res/spo_qualifier_results_format.txt"%(self.posp.file_prefix)
        df = self.res_file_to_df(filename).rename(columns={0:'s',1:'p',2:'o',3:'pq',4:'oq'}).reset_index(drop=True)

        re_pattern = re.compile("(.*),([^,]+)")
        spoq_df = pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['s'])[0],axis=1).tolist(), columns=['s_token','spos']). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['p'])[0],axis=1).tolist(), columns=['p_token','ppos'])). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['o'])[0],axis=1).tolist(), columns=['o_token','opos'])). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['pq'])[0],axis=1).tolist(), columns=['pq_token','pqpos'])). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['oq'])[0],axis=1).tolist(), columns=['oq_token','oqpos'])). \
                join(pd.DataFrame(df[['idx','parse_ranking']]))

        # save ranked token stats
        # spoq_df = spoq_df.merge(counts_spoq, on=['spos','ppos','opos','pqpos','oqpos'])
        re_pattern = re.compile("(.*)-[0-9]+$")
        spoq_df['sword'] = spoq_df['s_token'].apply(lambda row: re.match(re_pattern,row)[1])
        spoq_df['pword'] = spoq_df['p_token'].apply(lambda row: re.match(re_pattern,row)[1])
        spoq_df['oword'] = spoq_df['o_token'].apply(lambda row: re.match(re_pattern,row)[1])
        spoq_df['pqword'] = spoq_df['pq_token'].apply(lambda row: re.match(re_pattern,row)[1])
        spoq_df['oqword'] = spoq_df['oq_token'].apply(lambda row: re.match(re_pattern,row)[1])
        spoq_df_stats = spoq_df.groupby(['sword'])[['s_token']].count().rename(columns={'s_token':'s_token_N'}). \
            join(spoq_df.groupby(['sword'])[['parse_ranking']].mean()).rename(columns={'parse_ranking':'s_parse_ranking'}). \
            merge(spoq_df.groupby(['pword'])[['s_token']].count().rename(columns={'s_token':'p_token_N'}), left_index=True,right_index=True,how='outer'). \
            join(spoq_df.groupby(['pword'])[['parse_ranking']].mean()).rename(columns={'parse_ranking':'p_parse_ranking'}). \
            merge(spoq_df.groupby(['oword'])[['s_token']].count().rename(columns={'s_token':'o_token_N'}), left_index=True,right_index=True,how='outer'). \
            join(spoq_df.groupby(['oword'])[['parse_ranking']].mean()).rename(columns={'parse_ranking':'o_parse_ranking'}). \
            merge(spoq_df.groupby(['pqword'])[['pq_token']].count().rename(columns={'pq_token':'pq_token_N'}), left_index=True,right_index=True,how='outer'). \
            join(spoq_df.groupby(['pqword'])[['parse_ranking']].mean()).rename(columns={'parse_ranking':'pq_parse_ranking'}). \
            merge(spoq_df.groupby(['oqword'])[['oq_token']].count().rename(columns={'oq_token':'oq_token_N'}), left_index=True,right_index=True,how='outer'). \
            join(spoq_df.groupby(['oqword'])[['parse_ranking']].mean()).rename(columns={'parse_ranking':'oq_parse_ranking'}). \
            fillna(0.0)
        cols = ['s','p','o','pq','oq']
        stat_cols = ['s_token_N','p_token_N','o_token_N','pq_token_N','oq_token_N']
        spoq_df_stats['sum_N'] = spoq_df_stats[[c+'_token_N' for c in cols]].apply(sum,axis=1)
        spoq_df_stats = spoq_df_stats.sort_values(by='sum_N',ascending=False)
        spoq_df_stats.index.name="token"

        for col in cols:
            spoq_df_stats[col+'_token_score'] = spoq_df_stats[col+'_token_N'] / spoq_df_stats['sum_N']
        spoq_df_stats = self.apply_parse_ranking(
            df=spoq_df_stats, 
            cols=[c+'_token_score' for c in cols], 
            rank_cols=[c+'_parse_ranking' for c in cols])

        spoq_df_stats.to_csv(stats_directory+"ranked_spo_qualifier_tokens.csv",index=True)

        

    def rank_spo_qualifier_roles_types_hypothesis(self):
        stats_directory = "output/%s/stats/"%self.posp.file_prefix
        _ = os.makedirs(stats_directory) if not os.path.exists(stats_directory) else None

        # load data
        filename = "output/%s/res/spo_qualifier_results_format.txt"%(self.posp.file_prefix)
        df = self.res_file_to_df(filename).rename(columns={0:'s',1:'p',2:'o',3:'pq',4:'oq'}).reset_index(drop=True)

        re_pattern = re.compile("(.*),([^,]+)")
        spoq_df = pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['s'])[0],axis=1).tolist(), columns=['s_token','spos']). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['p'])[0],axis=1).tolist(), columns=['p_token','ppos'])). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['o'])[0],axis=1).tolist(), columns=['o_token','opos'])). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['pq'])[0],axis=1).tolist(), columns=['pq_token','pqpos'])). \
                join(pd.DataFrame(df.apply(lambda row: re.findall(re_pattern,row['oq'])[0],axis=1).tolist(), columns=['oq_token','oqpos'])). \
                join(pd.DataFrame(df[['idx','parse_ranking']]))
        spoq_df.index.name = 'org_position'
        spoq_df = spoq_df.reset_index(drop=False)

        # save roles for program, service, client relationships
        # save as program [offers] client [with] service
        predicates = ['provides-','offer-','provide-','provided-','offered-','include-']
        tmp1 = spoq_df[(spoq_df['pqpos']=='IN') &
                      (spoq_df['pq_token'].str.contains('with-')) &
                      (spoq_df['p_token'].str.contains('|'.join(predicates)))
                      ][['idx','parse_ranking','org_position','s_token','p_token','o_token','pq_token','oq_token']]
        tmp1.columns = ['idx','parse_ranking','org_position','program','action','client','direction','service']

        # save as program [offers] service [to,for] client
        tmp2 = spoq_df[(spoq_df['pqpos']=='IN') &
                      (spoq_df['pq_token'].str.contains('to-|for-'))&
                      (spoq_df['p_token'].str.contains('|'.join(predicates)))
                      ][['idx','parse_ranking','org_position','s_token','p_token','o_token','pq_token','oq_token']]
        tmp2.columns = ['idx','parse_ranking','org_position','program','action','service','direction','client']
        
        # save other scenarios
        roles_other = spoq_df.loc[~spoq_df.index.isin(list(set(tmp1.index.tolist()+tmp2.index.tolist())))][['idx','parse_ranking','org_position','s_token','p_token','o_token','pq_token','oq_token']]
        roles_other.columns = ['idx','parse_ranking','org_position','subject','action','object','qualifier_predicate','qualifier_object']

        roles = tmp1.append(tmp2)
        roles.sort_values(by='org_position').to_csv(stats_directory+"spo_qualifier_roles_prog_service_client.csv",index=False)
        # roles_other = roles_other#[['idx','program','action','service','direction','client']]
        roles_other.sort_values(by='org_position').to_csv(stats_directory+"spo_qualifier_roles_others.csv",index=False)
        
        

    def rank_spo_qualifier_ngram_hypothesis(self):
        stats_directory = "output/%s/stats/"%self.posp.file_prefix
        _ = os.makedirs(stats_directory) if not os.path.exists(stats_directory) else None

        roles = pd.read_csv(stats_directory+"spo_qualifier_roles_prog_service_client.csv")
        roles.columns = ['idx','parse_ranking','org_position','s_token','p_token','o_token','pq_token','oq_token']
        roles['format'] = 'psc'
        roles_other = pd.read_csv(stats_directory+"spo_qualifier_roles_others.csv")
        roles_other.columns = ['idx','parse_ranking','org_position','s_token','p_token','o_token','pq_token','oq_token']
        roles_other['format'] = 'spoq'
        spoq_df = roles.append(roles_other)

        re_pattern = re.compile("(.*)-[0-9]+$")
        spoq_df['sword'] = spoq_df['s_token'].apply(lambda row: re.match(re_pattern,row)[1])
        spoq_df['pword'] = spoq_df['p_token'].apply(lambda row: re.match(re_pattern,row)[1])
        spoq_df['oword'] = spoq_df['o_token'].apply(lambda row: re.match(re_pattern,row)[1])
        spoq_df['pqword'] = spoq_df['pq_token'].apply(lambda row: re.match(re_pattern,row)[1])
        spoq_df['oqword'] = spoq_df['oq_token'].apply(lambda row: re.match(re_pattern,row)[1])



        stat_cols = ['sword','pword','oword','pqword','oqword']
        tmp1 = spoq_df.groupby(stat_cols[0:1]).count()['idx'].sort_values().reset_index(drop=False)
        for ci in range(2,len(stat_cols)+1):
            tmp2 = spoq_df.groupby(stat_cols[0:ci]).count()['idx'].sort_values().reset_index(drop=False)
            merge_cols = list(set(tmp2.columns).intersection(tmp1.columns))
            merge_cols.remove('idx')
            tmp3 = tmp2.merge(tmp1, on=merge_cols)
            prob_col = 'prob(%s|%s)'%(stat_cols[ci-1],','.join(merge_cols))
            tmp3[prob_col] = tmp3['idx_x'] / tmp3['idx_y']
            tmp3.sort_values(by=['idx_y',prob_col],ascending=False).to_csv(stats_directory+'prob_spoq_%s_%s.csv'%(ci-1,stat_cols[ci-1]),index=False)
            tmp1 = tmp2


    def build_highest_spo_qualifier_phrases_hypothesis(self):
        stats_directory = "output/%s/stats/"%self.posp.file_prefix
        _ = os.makedirs(stats_directory) if not os.path.exists(stats_directory) else None
        phrases = self.ranked_phrases            

        # load data
        spoq_df_stats = pd.read_csv(stats_directory+"ranked_spo_qualifier_tokens.csv")
        roles = pd.read_csv(stats_directory+"spo_qualifier_roles_prog_service_client.csv")
        roles_other = pd.read_csv(stats_directory+"spo_qualifier_roles_others.csv")

        # save roles with highest phrase
        phrase_columns = [c for c in phrases.columns if type(c)==int]

        for c in phrase_columns:
            phrases["c_%s"%(c)] = phrases[c]

        res = []
        for id,grp in tqdm.tqdm(roles.groupby('idx')):
            idx = grp['idx'].iloc[0]
            phs = phrases[phrases['idx']==idx]
            for ri,row in grp.iterrows():
                # rank tokens in slot
                for val_col in ['program','service','client']:
                    val = row[val_col]
                token = val.split('-')[0]
                token_stats = spoq_df_stats[spoq_df_stats['token']==token].iloc[0]
                token_stats[['s_token_score','p_token_score','o_token_score','pq_token_score','oq_token_score']]
                # get highest ranked phrase for token
                for val_col in ['program','service','client']:
                    val = row[val_col]
                    cond = ' or '.join(["c_%s==\"%s\""%(c,val) for c in phrase_columns])
                    tmp = phs.query(cond).sort_values(by='score')
                    if len(tmp)>0:
                        label = tmp.loc[tmp['score'].idxmax()]['label']
                    else:
                        label = val.split('-')[0]
                    row[val_col] = label
                res.append(row)
        res = pd.concat(res,axis=1).T
        res.to_csv(stats_directory+"highest_ranked_spo_qualifier_phrases.csv",index=False)



    def spo_qualfiers_csv_to_output(self):
        stats_directory = "output/%s/stats/"%self.posp.file_prefix
        res = pd.read_csv(stats_directory+"highest_ranked_spo_qualifier_phrases.csv")
        if self.posp.program_names is not None:
            res = res.merge(self.posp.program_names, left_on=['idx'], right_index=True)

        prolog_roles = ":- style_check(-discontiguous).\n"
        for _,r in res.iterrows():
            prolog_roles += "%% %s\n"%(' '.join([str(v) for v in r.items()]))
            if 'Name' in r.keys():
                prolog_roles += "hasProgram(%s, \"%s\",\"%s\").\n"%(r['idx'],r['Name'],r['program'])
            prolog_roles += "offers(%s, \"%s\",\"%s\").\n"%(r['idx'],r['program'],r['service'])
            prolog_roles += "forClients(%s, \"%s\",\"%s\").\n"%(r['idx'],r['service'],r['client'])
            prolog_roles += "\n"
        filename = stats_directory+"spo_qualifiers_roles.pl"
        file = open(filename, "w")
        _=file.write(prolog_roles)
        file.close()


    def rank_by_length(self,series,df):
        tmp = pd.DataFrame(series.apply(lambda row: row.split(',')).tolist())
        # tmp = pd.DataFrame(series.apply(lambda row: sorted(row.split(','))))
        max_length = tmp.columns.shape[0]
        tmp[['idx','parse_ranking']] = df[['idx','parse_ranking']]
        tmp['text'] = series
        tmp['score'] = max_length - tmp.T.isna().sum()
        tmp['score'] = tmp['score']/tmp['score'].max()
        return tmp

    def spo_qualifier_pos_by_state_satsfier(self):

        stats_directory = "output/%s/stats/"%self.posp.file_prefix
        filename = "output/%s/res/spo_qualifier_results_format.txt"%(self.posp.file_prefix)
        df = self.res_file_to_df(filename).rename(columns={0:'s',1:'p',2:'o',3:'pq',4:'oq'}).reset_index(drop=True)
        df = df.merge(self.posp.sentences, on='idx',how='left')
        cols = ['s','p','o','pq','oq']
        cols_w = [c+'_word' for c in cols]
        cols_t = ['token_%s'%c for c in cols]
        cols_p = [c+'_pos' for c in cols]
        re_pattern1 = re.compile("(.*),([^,]+)")
        re_pattern2 = re.compile("(.*)-[0-9]+$")
        df = df.reset_index(drop=False).rename(columns={'index':'org_index'})
        for col in cols:
            df[col+'_word'] = df[col].apply(lambda text: re.search(re_pattern1, text)[1])
            df[col+'_word'] = df[col+'_word'].apply(lambda text: re.search(re_pattern2, text)[1])
            df[col+'_pos'] = df[col].apply(lambda text: re.search(re_pattern1, text)[2])
        tmp = df[['idx','parse_ranking','org_index','Text',cols_p[0],cols_w[0]]].merge(self.et.tokens[['token','category']], left_on=cols_w[0], right_on='token',how='left')
        tmp=tmp.rename(columns={'token':'token_s','category':'category_s'})
        tmp = tmp.merge(df[['org_index',cols_p[1],cols_w[1]]].merge(self.et.tokens[['token','category']], left_on=cols_w[1], right_on='token',how='left'), on='org_index')
        tmp=tmp.rename(columns={'token':'token_p','category':'category_p'})
        tmp = tmp.merge(df[['org_index',cols_p[2],cols_w[2]]].merge(self.et.tokens[['token','category']], left_on=cols_w[2], right_on='token',how='left'), on='org_index')
        tmp=tmp.rename(columns={'token':'token_o','category':'category_o'})
        tmp = tmp.merge(df[['org_index',cols_p[3],cols_w[3]]].merge(self.et.tokens[['token','category']], left_on=cols_w[3], right_on='token',how='left'), on='org_index')
        tmp=tmp.rename(columns={'token':'token_pq','category':'category_pq'})
        tmp = tmp.merge(df[['org_index',cols_p[4],cols_w[4]]].merge(self.et.tokens[['token','category']], left_on=cols_w[4], right_on='token',how='left'), on='org_index')
        tmp=tmp.rename(columns={'token':'token_oq','category':'category_oq'})
        # tmp = tmp.rename(columns = {'category':'category_4','token':'token_4'})
        tmp = tmp.dropna(axis=0, how='all',subset=cols_t)
        # tmp.to_csv('output/spoq_with_cat.csv')
#
        # combine stats into scores
        res = []
        cols_c = ['category_%s'%c for c in cols]
        for cc,c,p in zip(cols,cols_c,cols_p):
            tmp2 = tmp.groupby([c,p])[['idx']].count().rename(columns={'idx':'N'})
            tmp2['slot'] = cc
            res.append(tmp2)
        res_df = pd.concat(res).reset_index(drop=False)
        res_df.columns = ['category','pos','N', 'slot']
        tabs = res_df.pivot(index=['category','pos'], columns='slot',values='N')
        tabs = tabs.loc['satisfier'].merge(tabs.loc['state'], on='pos', how='outer', suffixes=['_sat','_st'])
        cols_tabs = np.array(list(zip(cols,[c+'_sat' for c in cols], [c+'_st' for c in cols])))
        missing_cols = set(flatten(cols_tabs)).difference(set(tabs.columns))
        for c in missing_cols:
            tabs[c] = np.nan
        totals = tabs[cols_tabs[:,1]].sum().sum()
        totals += tabs[cols_tabs[:,2]].sum().sum()
        for c,c1,c2 in cols_tabs:
            tabs[c] = tabs[c1] / (tabs[c1]+tabs[c2])
            tabs['p(st_'+c+')'] = tabs[c1] / totals
            tabs['p(sat_'+c+')'] = tabs[c2] / totals

        tabs.to_csv(stats_directory+'spo_quaifier_pos_by_state_satisfier.csv',index=True)

        tabs = tabs.fillna(0.0)
        plt.close()
        fig,ax = plt.subplots(1,2, figsize=(10,5))

        _=ax[0].set_title("P(state|pos,slot)")
        _=sns.heatmap(tabs[['p(st_'+c+')' for c in cols]], ax=ax[0],annot=True)
        _=ax[0].set_yticks(range(tabs.index.shape[0]))
        _=ax[0].set_yticklabels(tabs.index)

        _=ax[1].set_title("P(satisfier|pos,slot)")
        _=sns.heatmap(tabs[['p(sat_'+c+')' for c in cols]], ax=ax[1],annot=True)
        _=ax[1].set_yticks(range(tabs.index.shape[0]))
        _=ax[1].set_yticklabels(tabs.index)

        plt.tight_layout()
        plt.savefig(stats_directory+'spo_quaifier_pos_by_state_satisfier.pdf', bbox_inches="tight")
        plt.close()

    def gen_phrase_pos_rank(self):
        hp = self
        filename = "output/%s/res/phrase_pos_results_format.txt"%(hp.posp.file_prefix)
        df = hp.res_file_to_df(filename)
        val_cols = [c for c in df.columns if type(c) is int]
        res = []
        id_min = 0
        # id_min = vs.name
        for id,vs in tqdm.tqdm(df.loc[id_min:][['idx']+val_cols].iterrows(), total=df.loc[id_min:].shape[0]):
            idx=vs['idx']
            for v in vs[val_cols].values:
                if v==v:
                    match = re.search(r'(.*[\-[0-9]+),(.+)',v)
                    if match:
                        try:
                            token = match[1]
                            pos = match[2]
                            word = re.search(r'^(.+)\-[0-9]+$',token)[1]
                            res.append([id,idx,token,pos,word])
                        except TypeError:
                            pass
        res_df = pd.DataFrame(res, columns = ['phrase_id','idx','token','pos','word'])
        fileout = "output/%s/res/phrase_pos_results_words.csv"%(hp.posp.file_prefix)
        res_df.to_csv(fileout,index=False)

    def gen_spo_pos_df(self):
        hp = self
        filename = "output/%s/res/spo_results_format.txt"%(hp.posp.file_prefix)
        spo_df = hp.res_file_to_df(filename).rename(columns={0:'s',1:'p',2:'o'}).reset_index(drop=True)
        val_cols = ['s','p','o']
        res_spo = []
        id_min = 0
        

        # id_min = vs.name
        for id,vs in tqdm.tqdm(spo_df.loc[id_min:][['idx']+val_cols].iterrows(), total=spo_df.loc[id_min:].shape[0]):
            idx=vs['idx']
            for c,v in vs[val_cols].iteritems():
                if v==v:
                    match = re.search(r'(.*[\-[0-9]+),(.+)',v)
                    if match:
                        try:
                            token = match[1]
                            pos = match[2]
                            word = re.search(r'^(.+)\-[0-9]+$',token)[1]
                            res_spo.append([id,idx,c,token,pos,word])
                        except TypeError:
                            pass
        res_spo_df = pd.DataFrame(res_spo, columns = ['spo_id','idx','slot','token','pos','word'])
        return res_spo_df
    
    def gen_spoq_pos_df(self):
        hp = self
        filename = "output/%s/res/spo_qualifier_results_format.txt"%(hp.posp.file_prefix)
        df = hp.res_file_to_df(filename).rename(columns={0:'s',1:'p',2:'o',3:'pq',4:'oq'}).reset_index(drop=True)
        val_cols = ['s','p','o', 'pq','oq']
        res = []
        id_min = 0
        

        # id_min = vs.name
        for id,vs in tqdm.tqdm(df.loc[id_min:][['idx','parse_ranking']+val_cols].iterrows(), total=df.loc[id_min:].shape[0]):
            idx=vs['idx']
            parse_ranking = vs['parse_ranking']
            for c,v in vs[val_cols].iteritems():
                if v==v:
                    match = re.search(r'(.*[\-[0-9]+),(.+)',v)
                    if match:
                        try:
                            token = match[1]
                            pos = match[2]
                            word = re.search(r'^(.+)\-[0-9]+$',token)[1]
                            res.append([id,idx,c,token,pos,word, parse_ranking])
                        except TypeError:
                            pass
        res_spoq_df = pd.DataFrame(res, columns = ['spoq_id','idx','slot','token','pos','word', 'parse_ranking'])
        return res_spoq_df
    


    
    def phrase_WP_pos_ranking(self):
        hp = self
        res_spo_df = hp.gen_spo_pos_df()
        filename = "output/%s/res/spo_results_format.txt"%(hp.posp.file_prefix)
        spo_df = hp.res_file_to_df(filename).rename(columns={0:'s',1:'p',2:'o'}).reset_index(drop=True)

        fileout = "output/%s/res/phrase_pos_results_words.csv"%(hp.posp.file_prefix)
        res_df = pd.read_csv(fileout)
        # df_pos = res_df[res_df.pos=='WP']
        # df_pos = res_df[res_df.pos=='VBZ']
        # tmps = res_spo_df[res_spo_df.slot=='s'].merge(df_pos,on=['idx','token','word','pos']).drop(columns=['slot','word'])
        tmp = res_spo_df[res_spo_df.slot=='p'].drop(columns=['slot','word']).merge(tmps,on=['spo_id','idx'], suffixes=['_p','_s'])
        tmp = res_spo_df[res_spo_df.slot=='o'].drop(columns=['slot','word']).merge(tmp,on=['spo_id','idx'], suffixes=['_o','_s']).rename(columns={'token':'token_o', 'pos':'pos_o'})
        tmp2 = res_df.merge(tmp,left_on=['idx','token','pos'], right_on=['idx','token_o','pos_o'], suffixes=['_o','_s']).drop(columns=['token','pos','word'])
        tmp2 = tmp2.merge(found_phrases, left_on=['idx','phrase_id_s'], right_on=['idx','phrase_id']).drop(columns=['phrase_id'])
        tmp2 = tmp2.merge(found_phrases, left_on=['idx','phrase_id_o'], right_on=['idx','phrase_id'], suffixes=['_s','_o']).drop(columns=['phrase_id'])

        tmp2 = spo_df[['parse_ranking']].merge(tmp2, left_index=True, right_on=['spo_id'])

        state_vb = ['VBG','VBZ']
        tmp2['state_o'] = 0.0
        tmp2.loc[tmp2[tmp2.pos_p.isin(state_vb)].index, 'state_o'] = 1.0

        satisfier_vb = ['VB','VBP']
        tmp2['satisfier_o'] = 0.0
        tmp2.loc[tmp2[tmp2.pos_p.isin(satisfier_vb)].index, 'satisfier_o'] = 1.0


    def from_mods(self):
        hp = self
        stats_directory = "output/%s/stats/"%hp.posp.file_prefix
        _ = os.makedirs(stats_directory) if not os.path.exists(stats_directory) else None

        res_spo_df = hp.gen_spo_pos_df()
        filename = "output/%s/res/spo_results_format.txt"%(hp.posp.file_prefix)
        spo_df = hp.res_file_to_df(filename).rename(columns={0:'s',1:'p',2:'o'}).reset_index(drop=True)
        spo_df.drop(columns=[c for c in spo_df.columns if type(c) == int], inplace=True)
        # spo_df.merge(hp.posp.sentences, on='idx', how='left').to_csv("output/%s/res/spo_to_check.csv"%(hp.posp.file_prefix), index=False)
        res_spoq_df = hp.gen_spo_pos_df()
        filename = "output/%s/res/spo_qualifier_results_format.txt"%(hp.posp.file_prefix)
        spoq_df = hp.res_file_to_df(filename).rename(columns={0:'s',1:'p',2:'o', 3:'pq', 4:'oq'}).reset_index(drop=True)
        spoq_df.drop(columns=[c for c in spoq_df.columns if type(c) == int], inplace=True)
        re_pattern = re.compile("(.*-[0-9]+)")
        spoq_df['token_s'] = spoq_df['s'].apply(lambda row: re.match(re_pattern,row)[1])
        spoq_df['token_p'] = spoq_df['p'].apply(lambda row: re.match(re_pattern,row)[1])
        spoq_df['token_o'] = spoq_df['o'].apply(lambda row: re.match(re_pattern,row)[1])
        spoq_df['token_pq'] = spoq_df['pq'].apply(lambda row: re.match(re_pattern,row)[1])
        spoq_df['token_oq'] = spoq_df['oq'].apply(lambda row: re.match(re_pattern,row)[1])
        spoq_df.merge(hp.posp.sentences, on='idx', how='left').to_csv("output/%s/res/spoq_to_check.csv"%(hp.posp.file_prefix), index=False)

        fileout = "output/%s/res/phrase_pos_results_words.csv"%(hp.posp.file_prefix)
        res_df = pd.read_csv(fileout)



        found_phrases = []
        for ix,grp in tqdm.tqdm(res_df.groupby(['idx','phrase_id'])[['word']]):
            found_phrases.append(list(ix)+[' '.join([str(xx).strip() for xx in grp['word'].values])])
        found_phrases = pd.DataFrame(found_phrases, columns=['idx','phrase_id','parsed_phrase'])

        filename = "output/%s/res/phrase_results_format.txt"%(hp.posp.file_prefix)
        phrase_df = hp.res_file_to_df(filename).rename(columns={0:'phrase'}).reset_index(drop=True)
        phrase_df = hp.rank_by_length(series=phrase_df['phrase'], df=phrase_df)
        phrase_df = hp.apply_parse_ranking(df=phrase_df)[['idx','score']]





        # found_phrases = found_phrases.merge(phrase_df[['parse_ranking']], left_on=['phrase_id'], right_index=True, how='left')

        filename = "output/%s/res/mod_results_format.txt"%(hp.posp.file_prefix)
        mod_df = hp.res_file_to_df(filename).rename(columns={0:'term',1:'mod'}).reset_index(drop=True)
        val_cols = ['mod','term']
        res_mod = []
        id_min = 0
        # id_min = vs.name
        for id,vs in tqdm.tqdm(mod_df.loc[id_min:][['idx']+val_cols].iterrows(), total=mod_df.loc[id_min:].shape[0]):
            idx=vs['idx']
            res_tmp = []
            for c,v in vs[val_cols].iteritems():
                if v==v:
                    match = re.search(r'(.*[\-[0-9]+),(.+)',v)
                    if match:
                        try:
                            token = match[1]
                            pos = match[2]
                            # word = re.search(r'^(.+)\-[0-9]+$',token)[1]
                            res_tmp.append([token,pos])
                        except TypeError:
                            res_tmp.append(['',''])
            row = [id,idx]+flatten(res_tmp)
            res_mod.append(row)

        res_mod_df = pd.DataFrame(res_mod, columns = ['mod_id','idx','mod_token','mod_pos','term_token','term_pos'])




        mod_pos = '|'.join(['JJ','VB'])
        df_pos = res_mod_df[res_mod_df.mod_pos.str.contains(mod_pos)]
        tmp0 = res_df.merge(df_pos,left_on=['idx','token','pos'], right_on=['idx','term_token','term_pos'], how='left'). \
              append(res_df.merge(df_pos,left_on=['idx','token','pos'], right_on=['idx','mod_token','mod_pos'])). \
              drop(columns=['word']).rename(columns={'phrase_id':'phrase_id_mod'}). \
              drop_duplicates()
        
        # tmp = res_df.merge(tmp,left_on=['idx','token','pos'], right_on=['idx','token_o','pos_o'], suffixes=['_s','_o']).drop(columns=['token','pos','word']).rename(columns={'phrase_id':'phrase_id_o'})
        # tmp.merge(res_df, on=['idx','token','pos']).rename(columns={'phrase_id':''})
        tmp0o = res_spo_df[res_spo_df.slot=='o'].merge(tmp0,on=['idx','token','pos'],how='inner').drop(columns=['slot']).rename(columns={'phrase_id_mod':'phrase_id_o', 'word':'word_o'})
        tmp0s = res_spo_df[res_spo_df.slot=='s'].merge(tmp0,on=['idx','token','pos'],how='inner').drop(columns=['slot']).rename(columns={'phrase_id_mod':'phrase_id_s', 'word':'word_s'})
        tmp = tmp0s.merge(tmp0o, on=['idx','spo_id'], suffixes=['_s','_o'],how='inner')
        tmp = res_spo_df[res_spo_df.slot=='p'].drop(columns=['slot']).merge(tmp,on=['spo_id','idx']).rename(columns={'token':'token_p','pos':'pos_p','word':'word_p'})
        
        tmp2 = tmp.merge(found_phrases, left_on=['idx','phrase_id_o'], right_on=['idx','phrase_id']).drop(columns=['phrase_id'])
        tmp2 = tmp2.merge(found_phrases, left_on=['idx','phrase_id_s'], right_on=['idx','phrase_id'], suffixes=['_o','_s']).drop(columns=['phrase_id'])


        tmp2['mod_term_o'] = tmp2.apply(lambda row: '' if type(row['mod_token_o'])== float else re.sub(r'\-[0-9]+','',row['mod_token_o']) + ' ' + re.sub(r'\-[0-9]+','',row['term_token_o']),axis=1)
        tmp2['mod_term_s'] = tmp2.apply(lambda row: '' if type(row['mod_token_s'])== float else re.sub(r'\-[0-9]+','',row['mod_token_s']) + ' ' + re.sub(r'\-[0-9]+','',row['term_token_s']),axis=1)

        # posp.sentences[posp.sentences.idx.isin(tmp3[tmp3.mod_term=='adopted parents'].idx)].drop_duplicates(['Text']).iloc[0].Text


        cols_check = ['word_s','word_p','word_o', 'pos_s','pos_p','pos_o','parsed_phrase_s','parsed_phrase_o', 'mod_term_s', 'mod_pos_s','term_pos_s', 'mod_term_o', 'mod_pos_o','term_pos_o']

        c1 = 'mod_term'
        c2 = 'parsed_phrase_o'
        c3 = 'parsed_phrase_s'


        tmp3 = tmp2.drop_duplicates(['idx','parsed_phrase_s','parsed_phrase_o','word_p','mod_term_s','mod_term_o']).reset_index(drop=True)
        score_cols = ['need_satisfier_in_o_1',         'need_satisfier_in_o_2',         'need_satisfier_in_o_3',    'client_state_in_o_6',         'need_in_o_8',         'need_in_o_9',   'need_satisfier_in_o_11',         'need_satisfier_in_s_12',         'need_satisfier_in_o_13', 'current_state_in_s_14',         'service_description_in_o_15',         'need_satisfier_in_o_16',         'service_in_o_17',         'need_satisfier_description_in_o_18',         'service_description_in_o_19',         'desired_state_in_o_20',    'need_satisfier_in_o_22',         'service_description_in_o_23',         'service_description_in_o_24',         'need_satisfier_in_o_25',    'required_for_in_s_27',         'required_criteria_in_o_28',         'eligibile_criteria_in_s_29',         'eligibile_for_in_o_30',         'need_satisfier_in_o_31',         'need_satisfier_description_in_o_32',         'service_description_in_o_33',         'client_description_in_p_34',         'program_in_s_35',         'need_satisfier_in_o_36',   'service_description_in_s_41',         'client_description_in_o_42',         'service_description_in_s_43', 'client_description_in_o_43', 'program_in_s_44',         'need_satisfier_in_o_44',  'program_in_s_45', 'service_description_in_o_45',      'need_satisfier_in_s_46',         'client_in_o_47']
        

        tmp3[score_cols] = 0.0
        tmp3.to_csv('output/'+hp.posp.file_prefix+'/stats/tmp3_pre_assignment.csv',index=False)
        # tmp3 = pd.read_csv('output/'+hp.posp.file_prefix+'/stats/tmp3_pre_assignment.csv')

        tmp3 = hp.assign_rule_score_obj(tmp3)
        # tmp3 = assign_rule_score_subj(tmp3)



        # add parse ranking for SPOs
        tmp3 = tmp3.merge(spo_df[['parse_ranking']], left_on=['spo_id'],right_index=True, how='left').rename(columns={'parse_ranking':'spo_parse_ranking'})
        # add parse ranking for SPOQs
        tmp3s = tmp3.merge(spoq_df[['idx','token_s','parse_ranking']], left_on=['idx','token_s'],right_on=['idx','token_s'], how='left'). \
            rename(columns={'parse_ranking':'spoq_parse_ranking_left'}). \
            fillna({'spoq_parse_ranking_left':0.0})
        tmp3 = tmp3.merge(tmp3s.groupby(['idx','parsed_phrase_s','parsed_phrase_o','word_p','mod_term_s','mod_term_o'])['spoq_parse_ranking_left'].mean().reset_index(drop=False),
            on=['idx','parsed_phrase_s','parsed_phrase_o','word_p','mod_term_s','mod_term_o'],how='left')
        tmp3o = tmp3.merge(spoq_df[['idx','token_o','parse_ranking']], left_on=['idx','token_s'],right_on=['idx','token_o'], how='left', suffixes=['','_to_delete']). \
            rename(columns={'parse_ranking':'spoq_parse_ranking_right'}). \
            drop(columns={'token_o_to_delete'}). \
            fillna({'spoq_parse_ranking_right':0.0})
        tmp3 = tmp3.merge(tmp3o.groupby(['idx','parsed_phrase_s','parsed_phrase_o','word_p','mod_term_s','mod_term_o'])['spoq_parse_ranking_right'].mean().reset_index(drop=False),
            on=['idx','parsed_phrase_s','parsed_phrase_o','word_p','mod_term_s','mod_term_o'],how='left')
        # add parse ranking for subject phrase
        tmp3 = tmp3.merge(phrase_df[['score']], left_on=['phrase_id_s'],right_index=True, how='left').rename(columns={'score':'phrase_parse_ranking_s'})
        # add parse ranking for object phrase
        tmp3 = tmp3.merge(phrase_df[['score']], left_on=['phrase_id_o'],right_index=True, how='left').rename(columns={'score':'phrase_parse_ranking_o'})

        tmp3.to_csv('output/'+hp.posp.file_prefix+'/stats/tmp3_obj.csv',index=False)
        tmp3 = pd.read_csv('output/'+hp.posp.file_prefix+'/stats/tmp3_obj.csv')


        # for c in [c for c in tmp3.columns if 'parse_rank' in c and '_norm' not in c]:
        #     tmp3[c+'_norm'] = tmp3[c] / tmp3[c].max()


        key_cols = [[c,re.sub('_in_[a-z](_[0-9]+)$','',c), re.sub('.*_in_([a-z])_[0-9]+$',r'\1',c)] for c in score_cols]
        key_cols = [k+[k[1]] if not 'description' in k[1] else k+[k[1].replace('_description','')] for k in key_cols]

        sum_cols = list(set([(k[1],k[2],'pos_score') for k in key_cols]))
        tmp3[sum_cols] = 0.0
   

        # ranked_val_cols = flatten(list(zip([c+'_val' for c in sum_cols], [c+'_ranked' for c in sum_cols])))
        # keep_cols = ['idx','token_s', 'token_p', 'token_o', 'pos_s', 'pos_p', 'pos_o', 'parsed_phrase_s', 'word_p', 'parsed_phrase_o']  + ranked_val_cols
        # tmp3=tmp3[tmp3.parsed_phrase_s=='Calgary Clubs Girls']



        # tmp3 = tmp3[(tmp3.program_in_s_35>0)&(tmp3.need_satisfier_in_o_36>0)]
        ranked_list = []
        for c,k,field,desc in tqdm.tqdm(key_cols):
            tmp = tmp3.copy()
            tmp['ranked_cat'] = k
            tmp['ranked_slot'] = field
            tmp['ranked_token'] = tmp['token_'+field]


            if field == 'p':
                tmp['ranked_value'] = tmp['word_p']
                tmp['phrase_parse_ranking_'+field] = 1.0
            else: # s and o
                tmp['ranked_value'] = tmp['parsed_phrase_'+field]

            # eq 1: parsed ranking
            tmp[('parse_ranking_score')] = tmp[['spo_parse_ranking','phrase_parse_ranking_'+field, 'spoq_parse_ranking_left', 'spoq_parse_ranking_right']].mean(axis=1)

            # eq 2: pos ranking
            # tmp[(k,field,'pos_score')] += tmp[c]
            keep_cols = ['idx','ranked_token','ranked_cat','ranked_slot','ranked_value','parse_ranking_score',c]
            ranked_list.append(tmp[keep_cols].copy())


            
        ranked_df = pd.concat(ranked_list).fillna(0.0)



        # ranked_df = ranked_df.drop_duplicates()

        ranked_df = ranked_df.merge(ranked_df.groupby(['idx','ranked_value','ranked_token','ranked_cat']).sum()[score_cols].sum(axis=1).reset_index(), on=['idx','ranked_value','ranked_token','ranked_cat'],how='left').rename(columns={0:'cat_sum'})
        ranked_df = ranked_df.merge(ranked_df.groupby(['idx','ranked_value','ranked_token']).sum()[score_cols].sum(axis=1).reset_index(), on=['idx','ranked_value','ranked_token'],how='left').rename(columns={0:'value_total'})
        ranked_df['pos_ranking'] = ranked_df['cat_sum'] / ranked_df['value_total']
        ranked_df['pos_ranking'] = ranked_df['pos_ranking']/ranked_df['pos_ranking'].max()

        ranked_df2 = ranked_df[['idx','ranked_value','ranked_token','ranked_cat','ranked_slot','parse_ranking_score','pos_ranking']].copy()
        ranked_df2['parsed_and_pos_ranking'] = ranked_df['parse_ranking_score'] * ranked_df2['pos_ranking']
        ranked_df2['parsed_and_pos_ranking'] = ranked_df2['parsed_and_pos_ranking'] / ranked_df2['parsed_and_pos_ranking'].max()
        ranked_df2.drop_duplicates(inplace=True)


        ranked_df2 = ranked_df2.merge(hp.posp.sentences, on=['idx'],how='left')
        # np.histogram(ranked_df.ranked_val.value_counts(),bins=200)[1][10]
        for k,grp in tqdm.tqdm(ranked_df2[ranked_df2.parsed_and_pos_ranking>0].groupby(['ranked_cat'])):
            # counts = grp.ranked_value.value_counts()
            # max_score = counts.quantile(.999)
            # terms = counts[counts<=max_score].index
            # grp2 = grp[grp.ranked_val.isin(terms)]
            grp2 = grp.copy().sort_values(by=['parsed_and_pos_ranking','idx','ranked_value','ranked_token'],ascending=[False,True,True,True])
            # grp2.ranked_score = grp2.ranked_score / grp2.ranked_score.max()
            grp2.to_csv('output/'+hp.posp.file_prefix+'/stats/tmp3_obj_ranked2_%s.csv'%(k),index=False)



        # # np.histogram(ranked_df.ranked_val.value_counts(),bins=200)[1][10]
        # for k,grp in tqdm.tqdm(ranked_df2[ranked_df2.parsed_and_pos_ranking>0].groupby(['ranked_cat'])):
        #     counts = grp.ranked_value.value_counts()
        #     max_score = counts.quantile(.999)
        #     terms = counts[counts<=max_score].index
        #     # grp2 = grp[grp.ranked_val.isin(terms)]
        #     grp2 = grp.copy().sort_values(by=['parsed_and_pos_ranking','idx','ranked_value'],ascending=[False,True,True])
        #     # grp2.ranked_score = grp2.ranked_score / grp2.ranked_score.max()
        #     grp2.to_csv('output/'+hp.posp.file_prefix+'/stats/tmp3_obj_ranked2_%s.csv'%(k),index=False)

        with pd.ExcelWriter('output/'+hp.posp.file_prefix+'/stats/ranked_entities_%s_scores.xlsx'%(hp.run_label), engine='xlsxwriter') as writer:  
            for k,grp in tqdm.tqdm(ranked_df2[ranked_df2.parsed_and_pos_ranking>0].groupby(['ranked_cat'])):
                grp2 = grp[['idx', 'ranked_value', 'ranked_slot', 'parsed_and_pos_ranking','pos_ranking','parse_ranking_score', 'Text']].copy().sort_values(by=['parsed_and_pos_ranking','idx','ranked_value'],ascending=[False,True,True])
                grp2.drop_duplicates(['idx','ranked_value','ranked_slot'],keep='first', inplace=True)
                grp2.to_excel(writer, sheet_name=k, index=False)  

        with pd.ExcelWriter('output/'+hp.posp.file_prefix+'/stats/ranked_entities_%s.xlsx'%(hp.run_label),engine='xlsxwriter') as writer:  
            for k,grp in tqdm.tqdm(ranked_df2[ranked_df2.parsed_and_pos_ranking>0].groupby(['ranked_cat'])):
                grp2 = grp[['idx', 'ranked_value', 'ranked_slot', 'parsed_and_pos_ranking', 'Text']].copy().sort_values(by=['parsed_and_pos_ranking','idx','ranked_value'],ascending=[False,True,True])
                grp2.drop_duplicates(['idx','ranked_value','ranked_slot'],keep='first', inplace=True)
                grp2.to_excel(writer, sheet_name=k, index=False)  

        with pd.ExcelWriter('output/'+hp.posp.file_prefix+'/stats/ranked_entities_%s_check.xlsx'%(hp.run_label),engine='xlsxwriter') as writer:  
            for k,grp in tqdm.tqdm(ranked_df2[ranked_df2.parsed_and_pos_ranking>0].groupby(['ranked_cat'])):
                grp2 = grp[['idx', 'ranked_value', 'ranked_token','ranked_slot', 'parsed_and_pos_ranking', 'Text']].copy().sort_values(by=['parsed_and_pos_ranking','idx','ranked_value'],ascending=[False,True,True])
                grp2.drop_duplicates(['idx','ranked_value','ranked_slot'],keep='first', inplace=True)
                grp2.to_excel(writer, sheet_name=k, index=False)  



    def analysis_graphs():
        plt.close()
        ranked_df3 = ranked_df2.copy()
        unique_cat = ranked_df2['ranked_cat'].unique()
        unique_cat.sort()
        fig,ax = plt.subplots(len(unique_cat)//4,5, figsize=(15,10))
        fig.suptitle("Scores for pos_ranking vs parse_ranking_score")
        xy = np.resize(unique_cat, ax.shape)
        for k,grp in ranked_df3.groupby(['ranked_cat']):
            xi,xj = np.where(xy==k)
            if len(xi)==0:
                continue
            xi=xi[0]
            xj=xj[0]
            ax[xi][xj].set_title(k)
            grp['pos_ranking'].hist(bins=20,ax=ax[xi][xj],color='r',alpha=0.5)
            grp['parse_ranking_score'].hist(bins=20,ax=ax[xi][xj],color='g',alpha=0.5)
            ax[xi][xj].legend([plt.Line2D([0], [0], color='r', lw=4), plt.Line2D([0], [0], color='g', lw=4)], ['pos','parsed'])

        plt.tight_layout()
        plt.savefig(savedir+'pos_vs_parse_hist.pdf', bbox_inches="tight")
        plt.close()


        ranked_df3 = ranked_df2[ranked_df2.parsed_and_pos_ranking>0].copy()
        unique_cat = ranked_df2['ranked_cat'].unique()
        unique_cat.sort()
        plt.close()
        fig,ax = plt.subplots(len(unique_cat)//4,5, figsize=(15,10))
        fig.suptitle("Score for %s"%('parsed_and_pos_ranking'))
        xy = np.resize(unique_cat, ax.shape)
        for k in unique_cat:
            grp = ranked_df3[ranked_df3.ranked_cat == k]
            xi,xj = np.where(xy==k)
            xi=xi[0]
            xj=xj[0]
            ax[xi][xj].set_title(k)
            if len(grp) > 0:
                grp['parsed_and_pos_ranking'].hist(bins=20,ax=ax[xi][xj],color='b',label='parsed_and_pos_ranking')
                ax[xi][xj].legend()#[plt.Line2D([0], [0], color='b', lw=4)], ['pos','parsed'])

        plt.tight_layout()
        plt.savefig(savedir+'pos_and_parse_hist.pdf', bbox_inches="tight")
        plt.close()

        ranked_df3 = ranked_df2.copy()#[ranked_df2.parsed_and_pos_ranking>0].copy()
        unique_cat = ranked_df2['ranked_cat'].unique()
        unique_cat.sort()
        plt.close()
        fig,ax = plt.subplots(len(unique_cat)//4,5, figsize=(15,10))
        fig.suptitle("Score for %s"%('parsed_and_pos_ranking'))
        xy = np.resize(unique_cat, ax.shape)
        for k in unique_cat:
            grp = ranked_df3[ranked_df3.ranked_cat == k]
            xi,xj = np.where(xy==k)
            xi=xi[0]
            xj=xj[0]
            ax[xi][xj].set_title(k)
            if len(grp) > 0:
                ax[xi][xj].plot(grp['parsed_and_pos_ranking'].sort_values(ascending=False).values,color='b',label='parsed_and_pos_ranking')

        plt.legend()
        plt.tight_layout()
        plt.savefig(savedir+'pos_and_parse_score.pdf', bbox_inches="tight")
        plt.close()




        cats = ranked_df2.ranked_cat.unique()
        need_cols = [c for c in cats if 'need' in c]
        client_cols = [c for c in cats if 'client' in c]
        service_cols = [c for c in cats if 'service' in c]

        tmp_df = ranked_df2[~ranked_df2['parsed_and_pos_ranking'].isnull()].pivot(columns='ranked_cat', values='pos_ranking')
        corr_df = tmp_df[cats].fillna(0.0).corr()
        sns.heatmap(corr_df, xticklabels=cats, yticklabels=cats)
        plt.show()

        ranked_df3 = ranked_df2.copy()
        nunique = ranked_df3['ranked_cat'].nunique()
        xy = np.resize(range(nunique), ax.shape)

        plt.close()
        fig,ax = plt.subplots(nunique//4,5, figsize=(15,10))
        for x,(k,grp) in tqdm.tqdm(enumerate(ranked_df3.groupby(['ranked_cat']))):
        # for x,(k,grp) in tqdm.tqdm(enumerate(ranked_df2.groupby(['ranked_cat']))):
            # break
            xi,xj = np.where(xy==x)
            if len(xi)==0:
                continue
            xi=xi[0]
            xj=xj[0]
            # if k=='client':
            #     break
            #      grp.sort_values(by=['parsed_and_pos_ranking'],ascending=False)[:50]
            ax[xi][xj].set_title(k)
            ax[xi][xj].set_xlabel('pos_ranking')
            ax[xi][xj].set_ylabel('parse_ranking_score')
            ax[xi][xj].scatter(grp['pos_ranking'], grp['parse_ranking_score'], s=1)

            # grp['parse_ranking_score'].hist(bins=20,ax=ax[xi][xj],color='b',alpha=0.5)
            # ax[xi][xj].legend([plt.Line2D([0], [0], color='g', lw=4), plt.Line2D([0], [0], color='r', lw=4)], ['pos','parsed'])

        plt.tight_layout()
        plt.show()
        plt.savefig(savedir+'pos_vs_parse_scatter.pdf', bbox_inches="tight")
        plt.close()

        savedir = "output/%s/reports/"%(hp.posp.file_prefix)
        _ = os.makedirs(savedir) if not os.path.exists(savedir) else None
        
        col = 'pos_ranking'
        col = 'parse_ranking_score'
        col = 'parsed_and_pos_ranking'
        random_state = 42
        ranked_df3 = ranked_df2[ranked_df2[col]>0].copy()
        for x,(k,grp) in tqdm.tqdm(enumerate(ranked_df3.groupby(['ranked_cat']))):
            grp2=grp.copy()
            grp2 = grp2.groupby(['ranked_value'])[[col]].mean()
            grp2['N'] = grp2.groupby(['ranked_value'])[col].count()
            # grp2['label'] = grp.apply(lambda row: "%s (N=row['ranked_value'])
            grp2 = grp2.sort_values(by=col,ascending=True).reset_index()
            grp2['rank'] = grp2[[col]].rank(pct=True)
            b1,b2,m1,m2,t1,t2 = grp2['rank'].quantile([.0,.10,.45,.55,.9,1.0])
            b2 = b2 if b2 < m1 else b2- 1e-10
            m2 = m2 if m2 < t1 else m2- 1e-10

            vals1 = grp2[grp2['rank'].between(b1,b2)]
            vals1 = vals1.sample(20,random_state=random_state) if vals1.shape[0]>= 20 else vals1
            vals1['c'] = 'r'
            vals2 = grp2[grp2['rank'].between(m1,m2)]
            vals2 = vals2.sample(20,random_state=random_state) if vals2.shape[0]>= 20 else vals2
            vals2['c'] = 'y'
            vals3 = grp2[grp2['rank'].between(t1,t2)]
            vals3 = vals3.sample(20,random_state=random_state) if vals3.shape[0]>= 20 else vals3
            vals3['c'] = 'g'
            vals = pd.concat([vals1,vals2,vals3]).sort_values(by=[col,'ranked_value'], ascending=[False,False])

            plt.close()
            fig,ax = plt.subplots(1, figsize=(5,8))
            ax.set_title("%s - Mean %s - to, mid, low"%(k,col))
            ax.set_xlabel('Mean %s'%(col))
            ax.set_ylabel('Entity')
            ax.invert_yaxis()
            plt.yticks(fontsize=7)
            ax.barh(vals.ranked_value,vals[col],label=k,color=vals.c)
            ax.legend([plt.Line2D([0], [0], color='g', lw=4), plt.Line2D([0], [0], color='y', lw=4), plt.Line2D([0], [0], color='r', lw=4)], ['high', 'mid', 'low'])
            plt.tight_layout()
            plt.savefig(savedir+'min_mid_top_%s_%s.pdf'%(k,col), bbox_inches="tight")
            plt.close()

        plt.legend()
        plt.show()
        # max_score = counts.quantile(.999)
        # terms = counts[counts>max_score].index
        # # grp2 = grp[grp.ranked_val.isin(terms)]
        grp2 = grp.copy().sort_values(by=['parsed_and_pos_ranking','idx','ranked_value'],ascending=[False,True,True])
        # grp2.ranked_score = grp2.ranked_score / grp2.ranked_score.max()
        # grp2.to_csv('output/'+hp.posp.file_prefix+'/stats/tmp3_obj_ranked2_%s.csv'%(k),index=False)



        # tmp3[keep_cols].to_csv('output/'+hp.posp.file_prefix+'/stats/tmp3_obj_ranked.csv',index=False)
        # plt.close()
        # tmp3[[c+'_ranked' for c in sum_cols]].hist()
        # plt.show()
        # tmp3[sum_cols]
        # plt.show()
        # # tmp3[[c[1]+'_score' for c in key_cols]] = 0.0
        # tmp3[[c[1]+'_score' for c in key_cols]] = 0.0
        # for c,k,field,desc in key_cols:
        #     t = tmp3[tmp3[c]>0]
        #     idx=t.index
        #     if 'p' in field:
        #         tmp3.loc[idx,c+'_score'] = (t[c+'_norm'] + t['spo_parse_ranking'])/2.0
        #     else:            
        #         tmp3.loc[idx,c+'_score'] = (t[c+'_norm'] + t['phrase_parse_ranking_'+field] + t['spo_parse_ranking'])/3.0


        # # sapling of wrods for evaluation

        # res = []
        # res_sample = []
        # for c in score_cols:
        #     t=tmp3[tmp3[c]==1]
        #     if t.shape[0]>0:
        #         t['cat']  = c
        #         # print(c, t.shape[0])
        #         # print(t.sample(min(t.shape[0],20))[['parsed_phrase_s','word_p','parsed_phrase_o']])
        #         tt = t[(~t.mod_pos_s.isnull())].copy()
        #         if tt.shape[0]==0:
        #             tt = t.copy()
        #         res_sample.append(tt.sample(min(tt.shape[0],20))[['idx','spo_id','cat','parsed_phrase_s','word_p','parsed_phrase_o','mod_term_s','mod_pos_s','term_pos_s']].copy())
        #         res.append(t[['idx','spo_id','parsed_phrase_s','word_p','parsed_phrase_o','cat']])
        # final_df = pd.concat(res)
        # final_df = final_df.merge(hp.posp.sentences, on='idx',how='left')
        # final2_df = pd.concat(res_sample)
        # final2_df = final2_df.merge(hp.posp.sentences, on='idx',how='left')
        # final2_df.to_csv('output/'+hp.posp.file_prefix+'/stats/tmp3_final2.csv',index=False)        


    def assign_rule_score_obj(self,tmp3_in):
        tmp3 = tmp3_in.copy()
        base_p = ['VB']
        past_p = ['VBD','VBN']
        present_p = ['VBG','VBP']
        present_3p = ['VBZ']
        in_p = ['IN']
        to_p_word = ['to','for']
        for_p_word = ['for']
        from_p_word = ['from','with']
        has_p_word = ['experience', 'struggle']
        own_p = ['PRP','PRP$']
        adj_pos = ['JJ']
        eq_adj_pos = ['JJS','JJR']
        noun_pos = ['NN']
        self_word = ['our','we','us']
        them_word = ['you','your','them','they','their', 'he','she'] + ['person','people','client','clients']
        # them_word = ['you','your','them','they','their', 'he','she'] + ['person','people','client','clients']

        # past
        t = tmp3[tmp3.pos_p=='VBP']
        # service in s
        # need_satisfiers ranking
        idx = t[(t.pos_s.isin(own_p))&(t.word_s.str.lower().isin(self_word))&(~t.pos_o.isin(['CD','DT']))&(t.mod_pos_o.isin(eq_adj_pos))].index
        tmp3.loc[idx,'need_satisfier_in_o_1'] += 1
        
        # need satisfiers
        # TODO: check p for direction
        idx = t[(t.pos_s.isin(own_p))&(t.word_s.str.lower().isin(self_word))&(~t.pos_o.isin(['CD','DT','JJ','PRP','NNP','NNPS','NNS']))&(t.mod_pos_o.isin(adj_pos))].index
        tmp3.loc[idx,'need_satisfier_in_o_2'] += 1

        idx = t[(t.pos_s.isin(own_p))&(t.word_s.str.lower().isin(self_word))&(~t.pos_o.isin(['CD','DT','NN','PRP']))&(t.mod_pos_o.isin(adj_pos))].index
        tmp3.loc[idx,'need_satisfier_in_o_3'] += 1

        # no good
        # idx = t[(t.pos_s.isin(own_p))&(t.word_s.str.lower().isin(self_word))&(t.pos_o.isin(['NNP']))&(~t.mod_pos_o.isin(adj_pos))].index
        # tmp3.loc[idx,'service_in_o_4'] += 1

        # no good
        # idx = t[(t.pos_s.isin(own_p))&(t.word_s.str.lower().isin(self_word))&(t.pos_o.isin(['NNP']))&(t.mod_pos_o.isin(['VBD','VBZ','VBP','VB']))].index
        # tmp3.loc[idx,'service_in_o_5'] += 1


        # client in s
        # idx = t[(t.pos_s.isin(own_p))&(t.word_s.str.lower().isin(them_word))&(t.mod_pos_o.isin(eq_adj_pos))].index
        # remove p =='need' and o=='information'
        # tt = t.loc[idx]
        # idx = tt[(tt.pos_p.str.lower().isin(['need']))&(tt.parsed_phrase_s.str.lower().isin(['information']))].index
        # tmp3.loc[idx,'client_state_in_o_6'] += 1


        # not good enough
        # idx = t[(t.pos_s.isin(own_p))&(t.word_s.str.lower().isin(them_word))&(t.mod_pos_o.isin(adj_pos))].index
        # tmp3.loc[idx,'need_satisfier_in_o_7'] += 1
        
        idx = t[(t.pos_s.isin(own_p))&(t.word_s.str.lower().isin(them_word))&(~t.mod_pos_o.isin(adj_pos+eq_adj_pos))].index
        tmp3.loc[idx,'need_in_o_8'] += 1
        # Todo: check direciton of p
        idx = t[(t.pos_s.isin(own_p))&(t.word_s.str.lower().isin(them_word))&(t.mod_pos_o.isin(adj_pos))].index
        tmp3.loc[idx,'need_in_o_9'] += 1


        # past - participle
        t = tmp3[tmp3.pos_p=='VBN']
        # service in s

        # no good: replaced by one below
        # 50/50 for needs and need satisfier
        # idx = t[(t.word_s.str.lower().isin(them_word))&(t.mod_pos_o.str.contains("JJ"))].index
        # tmp3.loc[idx,'need_satisfier_in_o_10'] += 1

        idx = t[(t.word_s.str.lower().isin(them_word))&(~t.mod_pos_o.isin(["VB","VBZ"]))].index
        tmp3.loc[idx,'need_satisfier_in_o_11'] += 1

        # client in s
        idx = t[(t.word_o.str.lower().isin(them_word))&(~t.mod_pos_s.isin(["VB","VBZ"]))].index
        tmp3.loc[idx,'need_satisfier_in_s_12'] += 1



        # present
        t = tmp3[tmp3.pos_p=='VBG']
        # service in o
        # idx = t[(t.word_s.str.lower().isin(self_word))].index
        # tmp3.loc[idx,'need_satisfier_in_o_13'] += 1
        # client in s
        idx = t[(t.pos_s.isin(own_p))&(t.word_s.str.lower().isin(them_word))].index
        tmp3.loc[idx,'current_state_in_s_14'] += 1

        # need in o
        idx = t[(t.pos_o.isin(['NN','NNS','NNP','NNPS']))].index
        tmp3.loc[idx,'need_satisfier_in_o_13'] += 1

        # present
        t = tmp3[tmp3.pos_p=='VBP']
        # service in s
        # n/a
        
        idx = t[(t.word_s.str.lower().isin(self_word))&(t.mod_pos_o=='VBG')&(~t.term_pos_o.isin(['DT','JJ','NNS','NNP','NNPS']))].index
        tmp3.loc[idx,'service_description_in_o_15'] += 1

        # TODO: check as some o are not corrrect (15/20 are good)
        idx = t[(t.word_s.str.lower().isin(self_word))&(t.mod_pos_o=='VBN')&(~t.term_pos_o.isin(['DT','NNPS','NNP']))].index
        tmp3.loc[idx,'need_satisfier_in_o_16'] += 1

        idx = t[(t.word_s.str.lower().isin(self_word))&(t.mod_pos_o=='VBN')&(t.term_pos_o.isin(['NNP','NNPS']))].index
        tmp3.loc[idx,'service_in_o_17'] += 1

        idx = t[(t.word_s.str.lower().isin(self_word))&(t.mod_pos_o=='VBZ')&(~t.term_pos_o.isin(['CD','NNP','NNS']))].index
        tmp3.loc[idx,'need_satisfier_description_in_o_18'] += 1

        idx = t[(t.word_s.str.lower().isin(self_word))&(t.mod_pos_o=='VBP')&(~t.term_pos_o.isin(['DT','NNPS','NNP']))].index
        tmp3.loc[idx,'service_description_in_o_19'] += 1

        # service in o
        # todo: combine with p + o to get full picture
        # todo: get p direction
        idx =  t[(t.word_s.str.lower().isin(them_word))&(~t.pos_o.isin(['NNP','NNPS','PRP','PRP$']))&(~t.mod_pos_o.isin(['JJ']))&(~t.term_pos_o.isin(['DT']))&(~t.pos_o.isin(['DT','WP','WDT']))].index
        tmp3.loc[idx,'desired_state_in_o_20'] += 1

        # no good.
        # idx = t[(t.word_s.str.lower().isin(them_word))&(t.mod_pos_o.isin(['JJS','JJR']))&(~t.term_pos_o.isin(['DT']))].index
        # tmp3.loc[idx,'client_state_in_o_21'] += 1
        
        # todo: check is s=client
        idx = t[(t.word_s.str.lower().isin(them_word))&(~t.mod_pos_o.isin(['JJ','JJS','JJR']))&(~t.term_pos_o.isin(['DT']))].index
        tmp3.loc[idx,'need_satisfier_in_o_22'] += 1
        

        # present
        t = tmp3[tmp3.pos_p=='VBZ']
        # service in s
        idx = t[(t.pos_s.isin(own_p))&(t.word_s.str.lower().isin(self_word))&(t.mod_pos_o.isin(['JJ','JJS','JJR']))&(~t.term_pos_o.isin(['DT']))].index
        tmp3.loc[idx,'service_description_in_o_23'] += 1
        idx = t[(t.pos_s.isin(own_p))&(t.word_s.str.lower().isin(self_word))&(~t.mod_pos_o.isin(['JJ','JJS','JJR']))&(t.term_pos_o.isin(['NNP']))].index
        tmp3.loc[idx,'service_description_in_o_24'] += 1

        # client in s
        idx = t[(t.pos_s.isin(own_p))&(t.word_s.str.lower().isin(them_word))&(t.pos_o=='NN')].index
        tmp3.loc[idx,'need_satisfier_in_o_25'] += 1

        # no food
        # idx = t[(t.word_s.str.lower().isin(them_word))&(t.word_s.str.lower().isin(them_word))&(~t.pos_o.isin(['NN','NNP','WP','WDT']))].index
        # tmp3.loc[idx,'need_satisfier_description_in_o_26'] += 1




        # TO -  too few 36 recrds

        # JJ
        t = tmp3[tmp3.pos_p.isin(['JJ'])]


        # todo: provide context, esp on s since s can be gathered form other service "description"
        # s=service, o=client
        idx = t[(t.mod_pos_o=='JJ')&(t.pos_s.isin(['NNS']))&(t.pos_o.isin(['NNS']))&(t.word_p=='available')].index
        tmp3.loc[idx,'required_for_in_s_27'] += 1
        tmp3.loc[idx,'required_criteria_in_o_28'] += 1


        # s=client, o=servicec
        idx = t[(t.mod_pos_o=='JJ')&(t.pos_s.isin(['NNS']))&(t.pos_o.isin(['NN','NNS']))&(t.word_p=='eligible')].index
        tmp3.loc[idx,'eligibile_criteria_in_s_29'] += 1
        tmp3.loc[idx,'eligibile_for_in_o_30'] += 1


        # NN
        t = tmp3[tmp3.pos_p.str.contains('NN')]


        idx = t[(t.pos_o.isin(['NN','NNS','NNP','NNPS']))&(t.mod_pos_o=='JJ')&(t.pos_s.isin(['NN','NNS']))].index
        tmp3.loc[idx,'need_satisfier_in_o_31'] += 1


        idx = t[(t.pos_o.isin(['NN','NNS','NNP','NNPS']))&(t.mod_pos_o.isin(['VBN']))&(t.pos_s.isin(['NNP','NNPS']))].index
        tmp3.loc[idx,'need_satisfier_description_in_o_32'] += 1

        # NNS
        t = tmp3[tmp3.pos_p.str.contains('NNS')]

        # p=client, o=client_state
        idx = t[(t.pos_s=='WP')&(t.mod_pos_o=='JJ')].index
        tmp3.loc[idx,'service_description_in_o_33'] += 1
        tmp3.loc[idx,'client_description_in_p_34'] += 1


        # program as s=NNP(S)
        t = tmp3[tmp3.pos_s.isin(['NNP','NNPS'])]

        offers_sym = ["provides", "offers", "offer", "provide", "provided", "offered", "offering",'include']


        idx = t[(t.word_p.isin(offers_sym))].index
        tmp3.loc[idx,'program_in_s_35'] += 1
        
        idx = t[(t.word_p.isin(offers_sym))&(t.pos_o.isin(['NN','NNS']))].index
        tmp3.loc[idx,'need_satisfier_in_o_36'] += 1



        # build spoq
        spoq_df = tmp3.reset_index(drop=False).merge(tmp3.reset_index(drop=False), left_on=['idx','token_o'], right_on=['idx','token_s'],how='inner', suffixes=['','q'])

        # save roles for program, service, client relationships
        # save as program [offers] client [with] service
        predicates = ['provides-','offer-','provide-','provided-','offered-','offering-','include-']
        spoq1 = spoq_df[(spoq_df['pos_pq']=='IN') &
                      (spoq_df['token_pq'].str.contains('with-')) &
                      (spoq_df['token_p'].str.contains('|'.join(predicates)))
                      ]
        spoq1[list(spoq1.columns[:11]) + list([c+'q' for c in spoq1.columns[4:11]])]
        # tmp1.columns = ['idx','parse_ranking','org_position','program','action','client','direction','service']
        # find SPO but don't flag Program as s when s"we" with pos_s=='PRP'
        
        # no good: 2 case only
        # idx = tmp3[tmp3.pos_s.isin(['NNP','NNPS'])].merge(spoq1[['index','idx','spo_id','phrase_id_s','phrase_id_o']], on=['idx','spo_id','phrase_id_s','phrase_id_o'],how='inner')['index']
        # tmp3.loc[idx,'program_in_s_39'] += 1

        # no good, only 2 cases
        # idx = tmp3[tmp3.pos_s.isin(['NNP','NNPS'])].merge(spoq1[['index','idx','spo_id','phrase_id_s','phrase_id_o']], on=['idx','spo_id','phrase_id_s','phrase_id_o'],how='inner')['index']
        # tmp3.loc[idx,'service_description_in_o_40'] += 1

        idx = tmp3[(~tmp3.pos_s.isin(['PRP']))&(tmp3.pos_o.isin(['NN','NNS','JJ','VBG','FW']))].merge(spoq1[['indexq','idx','spo_idq','phrase_id_sq','phrase_id_oq']], left_on=['idx','spo_id','phrase_id_s','phrase_id_o'],right_on=['idx','spo_idq','phrase_id_sq','phrase_id_oq'],how='inner')['indexq']
        tmp3.loc[idx,'service_description_in_s_41'] += 1
        tmp3.loc[idx,'client_description_in_o_42'] += 1

        idx = tmp3[tmp3.pos_o.isin(['NN','NNS','JJ'])].merge(spoq1[['indexq','idx','spo_idq','phrase_id_sq','phrase_id_oq']], left_on=['idx','spo_id','phrase_id_s','phrase_id_o'],right_on=['idx','spo_idq','phrase_id_sq','phrase_id_oq'],how='inner')['indexq']
        tmp3.loc[idx,'service_description_in_s_43'] += 1
        tmp3.loc[idx,'client_description_in_o_43'] += 1

        

        # save as program [offers] service [to,for] client
        spoq2 = spoq_df[(spoq_df['pos_pq']=='IN') &
                      (spoq_df['token_pq'].str.contains('to-|for-'))&
                      (spoq_df['token_p'].str.contains('|'.join(predicates)))
                      ]
        spoq2[list(spoq2.columns[:11]) + list([c+'q' for c in spoq2.columns[4:11]])]
        # tmp2.columns = ['idx','parse_ranking','org_position','program','action','service','direction','client']
        
        # find SPO but don't flag Program as s when s"we" with pos_s=='PRP'
        idx = tmp3[tmp3.pos_s.isin(['NNP','NNPS'])].merge(spoq2[['index','idx','spo_id','phrase_id_s','phrase_id_o']], on=['idx','spo_id','phrase_id_s','phrase_id_o'],how='inner')['index']
        tmp3.loc[idx,'program_in_s_44'] += 1
        idx = tmp3.merge(spoq2[['index','idx','spo_id','phrase_id_s','phrase_id_o']], on=['idx','spo_id','phrase_id_s','phrase_id_o'],how='inner')['index']
        tmp3.loc[idx,'need_satisfier_in_o_44'] += 1

        idx = tmp3[tmp3.pos_s.isin(['NN','NNS','JJ', 'JJ','VBG','FW'])].merge(spoq2[['index','idx','spo_id','phrase_id_s','phrase_id_o']], on=['idx','spo_id','phrase_id_s','phrase_id_o'],how='inner')['index']
        tmp3.loc[idx,'program_in_s_45'] += 1
        tmp3.loc[idx,'service_description_in_o_45'] += 1


        idx = tmp3.merge(spoq2[['indexq','idx','spo_idq','phrase_id_sq','phrase_id_oq']], left_on=['idx','spo_id','phrase_id_s','phrase_id_o'],right_on=['idx','spo_idq','phrase_id_sq','phrase_id_oq'],how='inner')['indexq']
        tmp3.loc[idx,'need_satisfier_in_s_46'] += 1
        tmp3.loc[idx,'client_in_o_47'] += 1

        return tmp3


    def merge_runs(self, merge_with='conj'):
        hp = self
        statsdir = 'output/'+hp.posp.file_prefix+'/stats/'
        savedir = 'output/'+hp.posp.file_prefix+'/reports/'
        org_sheets = ['program','service_description','required_criteria','need_satisfier','need_satisfier_description', 'client','client_description','desired_state','need']

        res = []
        for sheet1 in org_sheets:
            tmp = pd.read_csv(statsdir+'tmp3_obj_ranked2_%s.csv'%(sheet1))
            # if 'Evaluation' not in tmp.columns:
            #     tmp['']
            #     continue
            tmp['ranked_cat'] = sheet1
            res.append(tmp.copy())
        df0 = pd.concat(res)
        conjs_df = pd.read_csv(statsdir+"conjs_ranked.csv")
        corefs_df = pd.read_csv(statsdir+"corefs_ranked.csv")
        if merge_with == 'conj':
            tmp = df0.merge(conjs_df[['idx','ref','linked']], left_on=['idx','ranked_token'], right_on=['idx','ref'], how='inner'). \
                drop(columns=['ref','ranked_token','parse_ranking_score','pos_ranking','parsed_and_pos_ranking']). \
                rename(columns={'linked':'ranked_token', 'ranked_value':'org_ranked_value'})
        elif merge_with == 'coref':
            tmp = df0.merge(corefs_df[['idx','ref','coref']], left_on=['idx','ranked_token'], right_on=['idx','ref'], how='inner'). \
                drop(columns=['ref','ranked_token','parse_ranking_score','pos_ranking','parsed_and_pos_ranking']). \
                rename(columns={'coref':'ranked_token', 'ranked_value':'org_ranked_value'})
        else:
            tmp = df0.merge(conjs_df[['idx','ref','linked']], left_on=['idx','ranked_token'], right_on=['idx','ref'], how='inner'). \
                drop(columns=['ref','ranked_token','parse_ranking_score','pos_ranking','parsed_and_pos_ranking']). \
                rename(columns={'linked':'ranked_token', 'ranked_value':'org_ranked_value'})
            tmp = tmp.append(
                df0.merge(corefs_df[['idx','ref','coref']], left_on=['idx','ranked_token'], right_on=['idx','ref'], how='inner'). \
                    drop(columns=['ref','ranked_token','parse_ranking_score','pos_ranking','parsed_and_pos_ranking']). \
                    rename(columns={'coref':'ranked_token', 'ranked_value':'org_ranked_value'})
            )
        filename = "output/%s/res/phrase_results_format.txt"%(hp.posp.file_prefix)
        phrase_df = hp.res_file_to_df(filename).rename(columns={0:'phrase'}).reset_index(drop=True)
        phrase_df = hp.rank_by_length(series=phrase_df['phrase'], df=phrase_df)
        phrase_df = hp.apply_parse_ranking(df=phrase_df)[['idx','score']]

        fileout = "output/%s/res/phrase_pos_results_words.csv"%(hp.posp.file_prefix)
        res_df = pd.read_csv(fileout)
        found_phrases = []
        for ix,grp in tqdm.tqdm(res_df.groupby(['idx','phrase_id'])[['word']]):
            found_phrases.append(list(ix)+[' '.join([str(xx).strip() for xx in grp['word'].values])])
        found_phrases = pd.DataFrame(found_phrases, columns=['idx','phrase_id','parsed_phrase'])

        tmp = tmp.merge(res_df[['phrase_id','idx','token']], left_on=['idx','ranked_token'],right_on=['idx','token'], how='left')#.rename(columns={'score':'phrase_parse_ranking_s'})
        tmp = tmp.merge(found_phrases, left_on=['idx','phrase_id'],right_on=['idx','phrase_id'], how='left').rename(columns={'parsed_phrase':'ranked_phrase'})

        tmp = tmp.merge(phrase_df[['score']], left_on=['phrase_id'],right_index=True, how='inner').rename(columns={'score':'phrase_parse_ranking'}).drop(columns={'phrase_id'})

        tmp.to_csv(statsdir+'tmp3_obj_ranked2_all_%s.csv'%(merge_with),index=False)
        # tmp.to_csv(statsdir+'tmp3_obj_ranked2_all_coref.csv',index=False)



    def graph_merged_eval_data(self, combined_file='tmp3_obj_ranked2_all_conj.csv'):
        hp = self
        statsdir = 'output/'+hp.posp.file_prefix+'/stats/'
        savedir = 'output/'+hp.posp.file_prefix+'/reports/'
        _ = os.makedirs(savedir) if not os.path.exists(savedir) else None
        filepath = 'output/'+hp.posp.file_prefix+'/stats/ML Results - Evaluation_DR_2021-12-24.xlsx'
        sheets = ['program name','service_description', 'required_criteria','need_satisfier','need_satisfier_description','client demographic','client_description',
                'desired_state (outcome)','need']
        org_sheets = ['program','service_description','required_criteria','need_satisfier','need_satisfier_description', 'client','client_description','desired_state','need']

        res = []
        for sheet1,sheet2 in zip(org_sheets,sheets):
            tmp = pd.read_csv(statsdir+'tmp3_obj_ranked2_%s.csv'%(sheet1))
            # if 'Evaluation' not in tmp.columns:
            #     tmp['']
            #     continue
            tmp['ranked_cat'] = sheet2
            res.append(tmp.copy())
        org_df = pd.concat(res)
        org_df = org_df.groupby(['idx', 'ranked_value', 'ranked_token', 'ranked_cat', 'ranked_slot', 'Text']).mean().reset_index(drop=False)
        org_df2 = pd.read_csv(statsdir+combined_file)
        org_df2 = org_df2.groupby(['idx', 'org_ranked_value', 'ranked_token', 'ranked_phrase', 'ranked_cat', 'ranked_slot', 'Text']). \
            mean().reset_index(drop=False). \
            rename(columns={'ranked_phrase':'ranked_value'})

        xls = pd.ExcelFile(filepath)
        res = []
        for sheet in sheets:
            tmp = pd.read_excel(xls, sheet)
            if 'Evaluation' not in tmp.columns:
                tmp['only_Eval2'] = True
            #     continue
            tmp['ranked_cat'] = sheet
            res.append(tmp.copy())
        df = pd.concat(res)
        df = df[['idx','Eval2','only_Eval2','Evaluation','ranked_value']].merge(org_df[['idx','ranked_value','ranked_slot','parsed_and_pos_ranking','parse_ranking_score','pos_ranking','ranked_cat']], on=['idx','ranked_value'],how='outer')
        
        df2 = df[['idx','Eval2','only_Eval2','Evaluation','ranked_value','ranked_cat']].merge(org_df2[['idx','ranked_value','ranked_slot','ranked_cat','org_ranked_value']],left_on=['idx','ranked_value','ranked_cat'],right_on=['idx','org_ranked_value','ranked_cat'],how='inner', suffixes=['_old',''])
        df2 = df2.drop(columns=['ranked_value_old'])
        df = df.append(df2)
        # df = df[df.ranked_cat == 'need_satisfier']
        df.loc[df[(df.Evaluation=='partially correct')].index,'Evaluation'] = 'yes'
        df.loc[df[(df.Evaluation=='incorrect')].index,'Evaluation'] = 'maybe'
        df['Evaluation'] = df['Evaluation'].fillna('yes')

        hp.graph_evals(df=df)


        # # hardcoded analysis from Dec22 and Jan24 (coonj merge)
        stats = [['client demographic',416,424,.34,.33, 4723,.59],
                ['client_description',155,186,.54,.52, 155, .50],
                ['desired_state (outcome)',554, 915,.51,.56,1539,.12],
                ['need',549,1098,.59,.56, 1909,np.nan],
                ['need_satisfier',478,4062,.54,.78, 21201,.76],
                ['need_satisfier_description',173,193,.91,.94, 173,.66],
                ['program name',299,312,.82,.80, 6828,np.nan],
                ['required_criteria',142,212,.77,.80, 142,.23],
                ['service_description',535,2140,.93,.90, 3519,np.nan]]
        stats = pd.DataFrame(stats, columns=['entity', '1-level conjunctions', 'All Conjunctions','yes1','yes2', 'total','eval1_yes'])
        stats['no1'] = 1.0 - stats['yes1']
        stats['no2'] = 1.0 - stats['yes2']
        stats['eval1_no'] = 1.0 - stats['eval1_yes']
        stats['eval_yes'] = stats['eval1_yes'] + stats['yes1']*stats['eval1_no']
        stats['eval_no'] = 1.0 - stats['eval_yes']
        stats[['eval_yes', 'eval_no', 'eval1_yes','yes1', 'eval1_no']]

        stats['growth'] = (stats['All Conjunctions'] - stats['1-level conjunctions'])/stats['1-level conjunctions']
        stats['growth'] *= 100
        stats['yes'] = (stats['yes2'] - stats['yes1'])
        stats['yes'] *= 100
        stats['no'] = (stats['no2'] - stats['no1'])
        stats['no'] *= 100
        stats.index=[c.replace(' ','\n').replace('_','\n') for c in stats.entity]

        print('in change')
        plt.close()
        fig = plt.figure(figsize=(10,4))
        _=stats.growth.plot(kind='bar')
        plt.title('Eval Level-2\nEntity Growth % by adding conjunctions\n')
        # plt.title('X')
        plt.xlabel("Entity")
        plt.xticks(rotation=0)
        plt.ylabel("Growth %")
        plt.tight_layout()
        plt.savefig(savedir+'growth_conj_eval2.pdf', bbox_inches="tight")
        plt.close()

        plt.close()
        fig,ax = plt.subplots(1,figsize=(10,4))
        plt.plot(stats.index, [0]*stats.shape[0], linewidth=0.3, color='black')
        _ = stats[['yes','no']].plot.bar(rot=0,ax=ax,color=['green','red'])
        plt.title('Eval2\nCorrect/Incorrect Entity assignment change by adding conjunctions\n')
        plt.xlabel("Entity")
        plt.xticks(rotation=0)
        plt.ylabel("Yes/No Assignment %")
        plt.tight_layout()
        plt.savefig(savedir+'growth_conj_yes_no_eval2.pdf', bbox_inches="tight")
        plt.close()

    def review_annotations(self):
        hp = self
        savedir = 'output/'+hp.posp.file_prefix+'/reports/'
        filepath = 'output/'+hp.posp.file_prefix+'/stats/ML Results - Evaluation_DR_2021-12-24.xlsx'
        sheets = ['program name','service_description', 'required_criteria','need_satisfier','need_satisfier_description','client demographic','client_description',
                'desired_state (outcome)','need']
        org_sheets = ['program','service_description','required_criteria','need_satisfier','need_satisfier_description', 'client','client_description','desired_state','need']

        merged = True
        if merged:
            xls1 = pd.ExcelFile('output/'+hp.posp.file_prefix+'/stats/ranked_entities_Dec22_scores.xlsx')
            # xls1 = pd.ExcelFile('output/'+hp.posp.file_prefix+'/stats/ranked_entities_Jan24_scores.xlsx')
            res = []
            for sheet1,sheet2 in zip(org_sheets,sheets):
                tmp = pd.read_excel(xls1, sheet1)
                # if 'Evaluation' not in tmp.columns:
                #     tmp['']
                #     continue
                tmp['ranked_cat'] = sheet2
                res.append(tmp.copy())
            org_df = pd.concat(res)
        else:
            res = []
            for sheet1,sheet2 in zip(org_sheets,sheets):
                tmp = pd.read_csv('output/'+hp.posp.file_prefix+'/stats/tmp3_obj_ranked2_%s.csv'%(sheet1))
                tmp['ranked_cat'] = sheet2
                res.append(tmp)
            org_df = pd.concat(res)
            org_df.rename(columns={'ranked_field':'ranked_slot'},inplace=True)
        xls = pd.ExcelFile(filepath)
        res = []
        for sheet in sheets:
            tmp = pd.read_excel(xls, sheet)
            if 'Evaluation' not in tmp.columns:
                tmp['only_Eval2'] = True
            #     continue
            tmp['ranked_cat'] = sheet
            res.append(tmp.copy())
        df = pd.concat(res)
        if merged:
            df=df.merge(org_df[['idx','ranked_value','ranked_slot','parse_ranking_score','pos_ranking','ranked_cat']], on=['idx','ranked_value','ranked_slot','ranked_cat'],how='right')
        else:
            df=df.drop(columns=['parsed_and_pos_ranking']).merge(org_df[['idx','ranked_value','ranked_slot','parse_ranking_score','pos_ranking','parsed_and_pos_ranking','ranked_cat']], on=['idx','ranked_value','ranked_slot','ranked_cat'],how='right')

        df.loc[df[(df.Evaluation=='partially correct')].index,'Evaluation'] = 'yes'
        df.loc[df[(df.Evaluation=='incorrect')].index,'Evaluation'] = 'maybe'
        df['Evaluation'] = df['Evaluation'].fillna('yes')
        hp.graph_evals(df=df)

    # def merge_groups(df):

    def graph_evals(self,df):
        hp =self
        savedir = 'output/'+hp.posp.file_prefix+'/reports/'

        unique_cat = [cat for cat,grp in df.groupby(['ranked_cat'])]# if grp['BG-eval'].nunique() > 1]
        import random
        random.seed(42)
        color_tab = [plt.cm.tab20(i) for i in random.sample(range(20),20)]
        colors = dict(zip(df['Eval2'].unique().tolist()+df['Evaluation'].unique().tolist(), color_tab))
        # make sure 'yes', 'no', and 'maybe' have colours
        if 'yes' not in colors.keys():
            colors['yes'] = color_tab[len(colors.keys())]
        if 'no' not in colors.keys():
            colors['no'] = color_tab[len(colors.keys())]
        if 'maybe' not in colors.keys():
            colors['maybe'] = color_tab[len(colors.keys())]

        ###################################################################################
        # Graph creation
        ###################################################################################
        plt.close()
        fig,ax = plt.subplots(len(unique_cat)//3,3, figsize=(10,10))
        fig.suptitle("Count for Eval Level-1\n(* no Eval1 performed)")
        xy = np.resize(unique_cat, ax.shape)
        [axn.set_ylabel('N',rotation=0) for axn in ax[:,0]]
        [axn.set_xlabel('Eval1') for axn in ax[-1]]

        for k in unique_cat:
            grp = df[df.ranked_cat == k]
            xi,xj = np.where(xy==k)
            xi=xi[0]
            xj=xj[0]
            title_tag = '*' if grp.only_Eval2.sum()>0 else ''
            doc_count = grp.idx.nunique()
            ax[xi][xj].set_title(k+title_tag)
            ax[xi][xj].text(0.05,0.95,"Documents N=%s\nEntities N=%s"%(doc_count,grp.shape[0]),verticalalignment='top',transform=ax[xi][xj].transAxes,)
            if len(grp) > 0 and grp.Evaluation.nunique()>1:
                data = grp.Evaluation.value_counts()
                data = data.loc[data.index.sort_values(ascending=False)]
                data.plot(kind='bar',ax=ax[xi][xj],rot=0, color=[colors[e] for e in data.index], alpha=0.5)
                for p in ax[xi][xj].patches:
                    width = p.get_width()
                    height = p.get_height()
                    x, y = p.get_xy() 
                    ax[xi][xj].annotate(f'{(100*height/grp.shape[0]):.0f}%', (x + width/2, y + height*.9), ha='center')
                
        plt.tight_layout()
        plt.savefig(savedir+'evaluated_count_bar_eval1.pdf', bbox_inches="tight")
        plt.close()

        # plt.close()
        # fig,ax = plt.subplots(1, figsize=(10,10))
        # df.plot(kind='bar', ax=ax)
        # ax.text(0.05,0.95,"D N=30\nE N=10",verticalalignment='top',transform=ax.transAxes,)
        # plt.show()


        plt.close()
        fig,ax = plt.subplots(len(unique_cat)//3,3, figsize=(10,10))
        fig.suptitle("Count for Eval Level-2\n(on \"maybe\" subset)")
        xy = np.resize(unique_cat, ax.shape)
        [axn.set_ylabel('N',rotation=0) for axn in ax[:,0]]
        [axn.set_xlabel('Eval2') for axn in ax[-1]]
        for k in unique_cat:
            grp = df[(df['Eval2'] != 'y')&(df.ranked_cat == k)&(~df.Eval2.isnull())].copy()
            grp.loc[grp[(grp['Eval2'].str.startswith('y'))].index,'Eval2'] = 'yes'
            grp.loc[grp[(~grp['Eval2'].isin(['n','yes']))].index,'Eval2'] = 'no'
            grp.loc[grp[(grp['Eval2'] == 'n')].index,'Eval2'] = 'no'

            xi,xj = np.where(xy==k)
            xi=xi[0]
            xj=xj[0]
            doc_count = grp.idx.nunique()
            # ax[xi][xj].set_title(k+title_tag+"\nN=%s\nDocument N=%s"%(grp.shape[0], doc_count))
            ax[xi][xj].text(0.05,0.95,"Documents N=%s\nEntities N=%s"%(doc_count,grp.shape[0]),verticalalignment='top',transform=ax[xi][xj].transAxes,)
            ax[xi][xj].set_title(k)
            if len(grp) > 0 and grp.Eval2.nunique()>1:
                data = grp.Eval2.value_counts()
                data = data.loc[data.index.sort_values(ascending=False)]
                data.plot(kind='bar',ax=ax[xi][xj],rot=0, color=[colors[e] for e in data.index])
                for p in ax[xi][xj].patches:
                    width = p.get_width()
                    height = p.get_height()
                    x, y = p.get_xy() 
                    ax[xi][xj].annotate(f'{(100*height/grp.shape[0]):.0f}%', (x + width/2, y + height*1.02), ha='center')

        plt.tight_layout()
        plt.savefig(savedir+'evaluated_count_bar_eval2.pdf', bbox_inches="tight")
        plt.close()


        plt.close()
        fig,ax = plt.subplots(len(unique_cat)//3,3, figsize=(10,10))
        fig.suptitle("Count for Eval3\n(partial matches on \"maybe\" subset)\n(**Eval3 on entire dataset)")
        xy = np.resize(unique_cat, ax.shape)
        [axn.set_ylabel('N',rotation=0) for axn in ax[:,0]]
        [axn.set_xlabel('Eval3') for axn in ax[-1]]
        for k in unique_cat:
            grp1 = df[df.ranked_cat == k]
            grp = grp1[(grp1['Eval2'] != 'y')&(~grp1.Eval2.isnull())].copy()
            grp.loc[grp[(~grp['Eval2'].str.startswith('y'))].index,'Eval2'] = 'no'
            xi,xj = np.where(xy==k)
            xi=xi[0]
            xj=xj[0]
            
            title_tag = '**' if grp1['Evaluation'].count() == grp1['Eval2'].count() else ''
            ax[xi][xj].set_title(k+title_tag+"\nN=%s"%(grp.shape[0]))
            if len(grp) > 0:
                data = grp.Eval2.value_counts()
                data = data.loc[data.index.sort_values(ascending=False)]
                data.plot(kind='bar',ax=ax[xi][xj],rot=0, color=[colors[e] for e in data.index])
                for p in ax[xi][xj].patches:
                    width = p.get_width()
                    height = p.get_height()
                    x, y = p.get_xy() 
                    ax[xi][xj].annotate(f'{(100*height/grp.shape[0]):.0f}%', (x + width/2, y + height*1.02), ha='center')

        plt.tight_layout()
        plt.savefig(savedir+'evaluated_count_bar_hist_eval3.pdf', bbox_inches="tight")
        plt.close()


        ###################################################################################
        plt.close()
        fig,ax = plt.subplots(len(unique_cat)//3,3, figsize=(10,10))
        fig.suptitle("Approximate Count Eval1~Eval2: Eval1 adjusted with Eval2\n(* no Eval1 performed)\n(** Eval2 on entire dataset - no adjustments)")
        xy = np.resize(unique_cat, ax.shape)
        [axn.set_ylabel('N',rotation=0) for axn in ax[:,0]]
        [axn.set_xlabel('Eval1~Eval2') for axn in ax[-1]]

        for k in unique_cat:
            grp1 = df[df.ranked_cat == k]
            grp = df[(df['Eval2'] != 'y')&(df.ranked_cat == k)&(~df.Eval2.isnull())].copy()
            grp.loc[grp[(grp['Eval2'].str.startswith('y'))].index,'Eval2'] = 'yes'
            grp.loc[grp[(~grp['Eval2'].isin(['n','yes']))].index,'Eval2'] = 'no'
            grp.loc[grp[(grp['Eval2'] == 'n')].index,'Eval2'] = 'no'

            xi,xj = np.where(xy==k)
            xi=xi[0]
            xj=xj[0]
            if len(grp1) > 0:
                data1 = None
                if grp.only_Eval2.sum()==0:
                    title_tag = '**' if grp1.Evaluation.count() == grp.Eval2.count() else ''
                    data1 = grp1.Evaluation.value_counts()
                    if 'yes' not in data1.index:
                        data1.loc['yes'] = 0
                    if 'maybe' not in data1.index:
                        data1.loc['maybe'] = 0
                    data = grp.Eval2.value_counts().loc[['yes','no']]
                    data1.loc['yes'] = data1.loc['yes'] + int(data1.loc['maybe'] * data.loc['yes']/data.sum())
                    data1.loc['no'] = int(data1.loc['maybe'] * data.loc['no']/data.sum())
                    data1 = data1.loc[['yes','no']]
                else:
                    title_tag = '*' if grp1.Evaluation.count() != grp.Eval2.count() else ''
                    data1 = grp.Eval2.value_counts().loc[['yes','no']]
                ax[xi][xj].set_title(k+title_tag+"\nN=%s"%(data1.sum()))

                print(k+title_tag+"\nN=%s"%(data1.sum()))
                if data1 is not None:
                    data1 = data1.loc[['yes','no']]
                    data1.plot(kind='bar',ax=ax[xi][xj],rot=0, color=[colors[e] for e in data1.index])
                    for p in ax[xi][xj].patches:
                        width = p.get_width()
                        height = p.get_height()
                        x, y = p.get_xy() 
                        ax[xi][xj].annotate(f'{(100*height/data1.sum()):.0f}%', (x + width/2, y + height*1.02), ha='center')
                    # x = 3
        plt.tight_layout()
        plt.savefig(savedir+'evaluated_count_bar_adj_eval1.pdf', bbox_inches="tight")
        plt.close()



        ###################################################################################
        for col in ['parsed_and_pos_ranking','parse_ranking_score','pos_ranking']:
            plt.close()
            fig,ax = plt.subplots(len(unique_cat)//3,3, figsize=(10,10))
            fig.suptitle("Score for Eval1 vs %s\n(* no Eval1 performed)"%(col))
            xy = np.resize(unique_cat, ax.shape)
            [axn.set_ylabel('N',rotation=0) for axn in ax[:,0]]
            [axn.set_xlabel(col) for axn in ax[-1]]

            for k in unique_cat:
                grp = df[df.ranked_cat == k]
                # grp = grp[(grp['BG'].str.startswith('y')) | (grp['Evaluation']=='incorrect')]
                # grp = grp[grp.Evaluation=='incorrect']
                xi,xj = np.where(xy==k)
                xi=xi[0]
                xj=xj[0]
                title_tag = '*' if grp.Evaluation.nunique()<=1 else ''
                ax[xi][xj].set_title(k+title_tag+"\nN=%s"%(grp.shape[0]))
                if len(grp) > 0 and grp.Evaluation.nunique()>1:
                    for e,grp2 in grp.groupby(['Evaluation']):
                        grp2[col].hist(bins=10,ax=ax[xi][xj],color=colors[e],label="%s (N=%s)"%(e,grp2.shape[0]), alpha=0.5)
                    ax[xi][xj].legend()#[plt.Line2D([0], [0], color='b', lw=4)], ['pos','parsed'])

            plt.tight_layout()
            plt.savefig(savedir+'evaluated_%s_hist_eval1.pdf'%(col), bbox_inches="tight")
            plt.close()


            plt.close()
            fig,ax = plt.subplots(len(unique_cat)//3,3, figsize=(10,10))
            fig.suptitle("Score for Eval2 vs %s\n(on \"maybe\" subset)"%(col))
            xy = np.resize(unique_cat, ax.shape)
            [axn.set_ylabel('N',rotation=0) for axn in ax[:,0]]
            [axn.set_xlabel(col) for axn in ax[-1]]
            for k in unique_cat:
                grp = df[(df['Eval2'] != 'y')&(df.ranked_cat == k)&(~df.Eval2.isnull())].copy()
                grp.loc[grp[(grp['Eval2'].str.startswith('y'))].index,'Eval2'] = 'yes'
                grp.loc[grp[(~grp['Eval2'].isin(['n','yes']))].index,'Eval2'] = 'no'
                grp.loc[grp[(grp['Eval2'] == 'n')].index,'Eval2'] = 'no'
                xi,xj = np.where(xy==k)
                xi=xi[0]
                xj=xj[0]
                ax[xi][xj].set_title(k+"\nN=%s"%(grp.shape[0]))
                if len(grp) > 0:
                    for e,grp2 in grp.groupby(['Eval2']):
                        grp2[col].hist(bins=10,ax=ax[xi][xj],color=colors[e],label="%s (N=%s)"%(e,grp2.shape[0]), alpha=0.5)
                    ax[xi][xj].legend()#[plt.Line2D([0], [0], color='b', lw=4)], ['pos','parsed'])

            plt.tight_layout()
            plt.savefig(savedir+'evaluated_%s_hist_eval2.pdf'%(col), bbox_inches="tight")
            plt.close()


            plt.close()
            fig,ax = plt.subplots(len(unique_cat)//3,3, figsize=(10,10))
            fig.suptitle("Score for Eval3 vs %s\n(partial matches on \"maybe\" subset)"%(col))
            xy = np.resize(unique_cat, ax.shape)
            [axn.set_ylabel('N',rotation=0) for axn in ax[:,0]]
            [axn.set_xlabel(col) for axn in ax[-1]]
            for k in unique_cat:
                grp = df[(df['Eval2'] != 'y')&(df.ranked_cat == k)&(~df.Eval2.isnull())].copy()
                grp.loc[grp[(~grp['Eval2'].str.startswith('y'))].index,'Eval2'] = 'no'
                xi,xj = np.where(xy==k)
                xi=xi[0]
                xj=xj[0]
                ax[xi][xj].set_title(k+"\nN=%s"%(grp.shape[0]))
                if len(grp) > 0:
                    for e,grp2 in grp.groupby(['Eval2']):
                        grp2[col].hist(bins=10,ax=ax[xi][xj],color=colors[e],label="%s (N=%s)"%(e,grp2.shape[0]), alpha=0.5)
                    ax[xi][xj].legend()#[plt.Line2D([0], [0], color='b', lw=4)], ['pos','parsed'])

            plt.tight_layout()
            plt.savefig(savedir+'evaluated_%s_hist_eval3.pdf'%(col), bbox_inches="tight")
            plt.close()


        from sklearn import metrics
        for col in ['parsed_and_pos_ranking','parse_ranking_score','pos_ranking']:
            plt.close()
            fig,ax = plt.subplots(len(unique_cat)//3,3, figsize=(10,10))
            fig.suptitle("ROC for Eval1 vs %s\n(on \"maybe\" subset)\n(* no Eval1 performed)"%(col))
            xy = np.resize(unique_cat, ax.shape)
            for k in unique_cat:
                grp = df[df.ranked_cat == k]
                xi,xj = np.where(xy==k)
                xi=xi[0]
                xj=xj[0]
                title_tag = '*' if grp.only_Eval2.sum()>0 else ''
                ax[xi][xj].set_title(k+title_tag+"\nN=%s"%(grp.shape[0]))

                if grp.only_Eval2.sum()==0:
                    fpr, tpr, thresholds = metrics.roc_curve(grp.Evaluation, grp[col], pos_label='yes')
                    roc_auc = metrics.auc(fpr, tpr)

                    # Plot ROC curve
                    ax[xi][xj].plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
                    ax[xi][xj].plot([0, 1], [0, 1], 'k--')  # random predictions curve
                    ax[xi][xj].set_xlim([0.0, 1.0])
                    ax[xi][xj].set_ylim([0.0, 1.0])
                    ax[xi][xj].set_xlabel('False Positive Rate or (1 - Specifity)')
                    ax[xi][xj].set_ylabel('True Positive Rate or (Sensitivity)')
                    ax[xi][xj].legend(loc="lower right")

            plt.tight_layout()
            plt.savefig(savedir+'evaluated_%s_roc_eval1.pdf'%(col), bbox_inches="tight")
            plt.close()

        for col in ['parsed_and_pos_ranking','parse_ranking_score','pos_ranking']:
            plt.close()
            fig,ax = plt.subplots(len(unique_cat)//3,3, figsize=(10,10))
            fig.suptitle("ROC for Eval2 vs %s\n(on \"maybe\" subset)"%(col))
            xy = np.resize(unique_cat, ax.shape)
            for k in unique_cat:
                grp = df[(df['Eval2'] != 'y')&(df.ranked_cat == k)&(~df.Eval2.isnull())].copy()
                grp.loc[grp[(grp['Eval2'].str.startswith('y'))].index,'Eval2'] = 'yes'
                grp.loc[grp[(~grp['Eval2'].isin(['n','yes']))].index,'Eval2'] = 'no'
                grp.loc[grp[(grp['Eval2'] == 'n')].index,'Eval2'] = 'no'
                xi,xj = np.where(xy==k)
                xi=xi[0]
                xj=xj[0]
                ax[xi][xj].set_title(k+"\nN=%s"%(grp.shape[0]))

                fpr, tpr, thresholds = metrics.roc_curve(grp.Eval2, grp[col], pos_label='yes')
                roc_auc = metrics.auc(fpr, tpr)

                # Plot ROC curve
                ax[xi][xj].plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
                ax[xi][xj].plot([0, 1], [0, 1], 'k--')  # random predictions curve
                ax[xi][xj].set_xlim([0.0, 1.0])
                ax[xi][xj].set_ylim([0.0, 1.0])
                ax[xi][xj].set_xlabel('False Positive Rate or (1 - Specifity)')
                ax[xi][xj].set_ylabel('True Positive Rate or (Sensitivity)')
                ax[xi][xj].legend(loc="lower right")

            plt.tight_layout()
            plt.savefig(savedir+'evaluated_%s_roc_eval2.pdf'%(col), bbox_inches="tight")
            plt.close()

        #####################################################




class ShowResults:
    def __init__(self,posp):
        self.posp = posp
        self.pos_directory = 'output/%s/'%self.posp.file_prefix
        self.out_directory = 'output/%s/reports/'%self.posp.file_prefix
        self.results = {}

    def build_records(self):
        self.get_sentence_roots()
        self.get_coref_score()
        self.get_phrase_spo_score()
        self.get_spo_score()
        self.get_spo_qualifier_score()
        self.get_spo_qualifier_roles_score()

    def build_report(self,ids = None):
        self.parse_df(ids=ids)

    def get_sentence_roots(self):
        self.idx_sentence_starts = {}
        for filename in glob.glob(self.pos_directory+"data/*.pl"):
            try:
                idx =int(re.search(r'([0-9]+)\.pl',filename)[1])
            except:
                continue
            file = open(filename, "r")
            res = file.read()
            file.close()
            self.idx_sentence_starts[idx] = [int(t) for t in re.findall(r'gram\([0-9]+,"ROOT",\["ROOT\-([0-9]+)"',res)]


    def get_phrase_spo_score(self):
        stats_directory = self.pos_directory + "stats/"
        phrases_scores = pd.read_csv(stats_directory+'phrases_ranked.csv') 
        phrase_scores = phrases_scores[['idx','text', 'score']]
        phrases_scores['score(m)'] = phrases_scores['score']/phrases_scores['score'].max()
        self.phrases_scores = phrases_scores

    def get_coref_score(self):
        stats_directory = self.pos_directory + "stats/"
        corefs_scores = pd.read_csv(stats_directory+'corefs_ranked.csv') 
        # phrase_scores = phrases_scores[['idx','text', 'score']]
        corefs_scores['coref_score(i,t,cr)'] = corefs_scores['coref_score(i,t,cr)']/corefs_scores['coref_score(i,t,cr)'].max()
        self.corefs_scores = corefs_scores

    def get_spo_score(self):
        stats_directory = self.pos_directory + "stats/"
        spo_pos_score = pd.read_csv(stats_directory+'spo_ranked_by_pos_combination.csv',low_memory=False)
        spo_token_score = pd.read_csv(stats_directory+'spo_token_ranked.csv',low_memory=False)
        spo_pos_score['s_token_raw'] = spo_pos_score['s_token'].apply(lambda row: row.split('-')[0])
        spo_pos_score['p_token_raw'] = spo_pos_score['p_token'].apply(lambda row: row.split('-')[0])
        spo_pos_score['o_token_raw'] = spo_pos_score['o_token'].apply(lambda row: row.split('-')[0])
        spo_scores = spo_token_score.merge(spo_pos_score, right_on=['s_token_raw'],left_on='token',how='right')[['idx','token','s_token','s_token_score']]. \
             join(spo_token_score.merge(spo_pos_score, right_on=['p_token_raw'],left_on='token',how='right')[['p_token','p_token_score']]). \
             join(spo_token_score.merge(spo_pos_score, right_on=['o_token_raw'],left_on='token',how='right')[['o_token','o_token_score']])

        spo_scores['spo_token_score'] = spo_scores[['s_token_score','p_token_score','o_token_score']].mean(axis=1)
        spo_scores['spo_pos_score'] = spo_pos_score['score']
        spo_scores['spo_score(i)'] = spo_scores['spo_token_score'] * spo_scores['spo_pos_score']
        spo_scores['spo_score(i)'] = spo_scores['spo_score(i)']/spo_scores['spo_score(i)'].max()
        self.spo_scores = spo_scores


    def get_spo_qualifier_score(self):
        stats_directory = self.pos_directory + "stats/"

        spoq_pos_score = pd.read_csv(stats_directory+'spo_qualifier_ranked_by_pos_combination.csv')
        spoq_token_score = pd.read_csv(stats_directory+'spo_qualifier_token_ranked.csv')
        spoq_st_sat_score = pd.read_csv(stats_directory+'spo_quaifier_pos_by_state_satisfier.csv')

        spoq_pos_score['s_token_raw'] = spoq_pos_score['s_token'].apply(lambda row: row.split('-')[0])
        spoq_pos_score['p_token_raw'] = spoq_pos_score['p_token'].apply(lambda row: row.split('-')[0])
        spoq_pos_score['o_token_raw'] = spoq_pos_score['o_token'].apply(lambda row: row.split('-')[0])
        spoq_pos_score['pq_token_raw'] = spoq_pos_score['pq_token'].apply(lambda row: row.split('-')[0])
        spoq_pos_score['oq_token_raw'] = spoq_pos_score['oq_token'].apply(lambda row: row.split('-')[0])
        spoq_scores = spoq_token_score.merge(spoq_pos_score, right_on=['s_token_raw'],left_on='token',how='right')[['idx','token','s_token','s_token_score']]. \
             join(spoq_token_score.merge(spoq_pos_score, right_on=['p_token_raw'],left_on='token',how='right')[['p_token','p_token_score']]). \
             join(spoq_token_score.merge(spoq_pos_score, right_on=['o_token_raw'],left_on='token',how='right')[['o_token','o_token_score']]). \
             join(spoq_token_score.merge(spoq_pos_score, right_on=['pq_token_raw'],left_on='token',how='right')[['pq_token','pq_token_score']]). \
             join(spoq_token_score.merge(spoq_pos_score, right_on=['oq_token_raw'],left_on='token',how='right')[['oq_token','oq_token_score']])


        spoq_scores1 = spoq_st_sat_score.merge(spoq_pos_score, right_on=['spos'],left_on='pos',how='right')[['p(st_s)']]. \
             join(spoq_st_sat_score.merge(spoq_pos_score, right_on=['ppos'],left_on='pos',how='right')[['p(st_p)']]). \
             join(spoq_st_sat_score.merge(spoq_pos_score, right_on=['opos'],left_on='pos',how='right')[['p(st_o)']]). \
             join(spoq_st_sat_score.merge(spoq_pos_score, right_on=['pqpos'],left_on='pos',how='right')[['p(st_pq)']]). \
             join(spoq_st_sat_score.merge(spoq_pos_score, right_on=['oqpos'],left_on='pos',how='right')[['p(st_oq)']])


        spoq_scores2 = spoq_st_sat_score.merge(spoq_pos_score, right_on=['spos'],left_on='pos',how='right')[['p(sat_s)']]. \
             join(spoq_st_sat_score.merge(spoq_pos_score, right_on=['ppos'],left_on='pos',how='right')[['p(sat_p)']]). \
             join(spoq_st_sat_score.merge(spoq_pos_score, right_on=['opos'],left_on='pos',how='right')[['p(sat_o)']]). \
             join(spoq_st_sat_score.merge(spoq_pos_score, right_on=['pqpos'],left_on='pos',how='right')[['p(sat_pq)']]). \
             join(spoq_st_sat_score.merge(spoq_pos_score, right_on=['oqpos'],left_on='pos',how='right')[['p(sat_oq)']])


        spoq_scores = spoq_scores.join(spoq_scores1).join(spoq_scores2).fillna(0.0)


        spoq_scores['spoq_token_score'] = spoq_scores[['s_token_score','p_token_score','o_token_score','pq_token_score','oq_token_score']].mean(axis=1)
        spoq_scores['spoq_pos_score'] = spoq_pos_score['score']
        spoq_scores['spoq_state_score'] = spoq_scores[['p(st_s)','p(st_p)','p(st_o)','p(st_pq)','p(st_oq)']].mean(axis=1)
        spoq_scores['spoq_satisfier_score'] = spoq_scores[['p(sat_s)','p(sat_p)','p(sat_o)','p(sat_pq)','p(sat_oq)']].mean(axis=1)


        # spoq_scores['spoq_score(t,x)'] = spoq_scores['spoq_token_score']/spoq_scores['spoq_token_score'].max()
        spoq_scores['spoq_score_0(j)'] =  spoq_scores['spoq_token_score'] *           \
                                        spoq_scores['spoq_pos_score']
        spoq_scores['spoq_score(j)'] =  spoq_scores['spoq_token_score'] *           \
                                        spoq_scores['spoq_pos_score'] *             \
                                        spoq_scores['spoq_state_score'] *           \
                                        spoq_scores['spoq_satisfier_score']
        spoq_scores['spoq_score_0(j)'] = spoq_scores['spoq_score_0(j)']/spoq_scores['spoq_score_0(j)'].max()
        spoq_scores['spoq_score(j)'] = spoq_scores['spoq_score(j)']/spoq_scores['spoq_score(j)'].max()
        spoq_scores['p_n(sat_s)'] = spoq_scores['p(sat_s)']/spoq_scores['p(sat_s)'].max()
        # spoq_scores.to_csv('output/spoq_scores_1.csv')

        
        # plot_cols = ['spoq_score(j)','p(st_s)','p(st_p)','p(st_o)','p(st_pq)','p(st_oq)','p(sat_s)','p(sat_p)','p(sat_o)','p(sat_pq)','p(sat_oq)']
        # plt.close()
        # spoq_scores.sort_values(by=['p_n(sat_s)','spoq_score(j)']).reset_index()[['p_n(sat_s)','spoq_score(j)']].plot(style='.')
        # plt.show()
        # spoq_scores.merge(posp.sentences, on='idx', how='left').to_csv('output/spoq_scores.csv',index=False)
        self.spoq_scores = spoq_scores
        self.spoq_subset_scores()

    def spoq_subset_scores(self):
        spoq_scores = self.spoq_scores
        weights = [1,0.8,0.6,0.4,0.2]
        cols = ['p(st_s)','p(st_p)','p(st_o)','p(st_pq)','p(st_oq)']
        tqdm.tqdm.pandas()
        spoq_scores['s1'] = spoq_scores.progress_apply(lambda row: (weights*row[cols]).sum(),axis=1)
        cols = ['p(sat_s)','p(sat_p)','p(sat_o)','p(sat_pq)','p(sat_oq)']
        spoq_scores['s2'] = spoq_scores.progress_apply(lambda row: (weights*row[cols]).sum(),axis=1)
        self.spoq_scores = spoq_scores

    def get_spo_qualifier_roles_score(self):
        stats_directory = self.pos_directory + "stats/"
        
        spo_pos_score = pd.read_csv(stats_directory+'spo_qualifier_ranked_by_pos_combination.csv')

        roles_scores = pd.read_csv(stats_directory+'spo_qualifier_roles_prog_service_client.csv')
        roles_other_scores = pd.read_csv(stats_directory+'spo_qualifier_roles_others.csv')
        roles_scores['spoq_score(j)'] = self.spoq_scores.loc[roles_scores.org_position]['spoq_score(j)'].values
        self.roles_scores = roles_scores

        roles_other_scores = pd.read_csv(stats_directory+'spo_qualifier_roles_others.csv')
        roles_other_scores['spoq_score(j)'] = self.spoq_scores.loc[roles_other_scores.org_position]['spoq_score(j)'].values
        self.roles_other_scores = roles_other_scores

        # max_score = np.max([self.roles_scores['score'].max(),self.roles_other_scores['score'].max()])
        # self.roles_scores['spoq_pos_score(cq)'] = self.roles_scores['score']/max_score
        # self.roles_other_scores['spoq_pos_score(cq)'] = self.roles_other_scores['score']/max_score


    def parse_df(self,filename='output.txt',ids=None):
        _ = os.makedirs(self.out_directory) if not os.path.exists(self.out_directory) else None
        if ids is None:
            ids = list(self.idx_sentence_starts.keys())
        
        self.results = {}
        self.results_errors = []
        for idx in tqdm.tqdm(sorted(ids),total=len(ids)):
            self.results[idx] = {}
            self.format_data(idx=idx,label='Extracted Phrases Scores',df=self.phrases_scores,scores = ['score(m)'],cols=['text'])
            self.format_data(idx=idx,label='Extracted Corefs Scores',df=self.corefs_scores,scores = ['coref_score(i,t,cr)'],cols=['ref','coref'])
            self.format_data(idx=idx,label='Extracted SPO token scores',df=self.spo_scores,scores = ['spo_score(i)'],cols=['s_token','p_token','o_token'])
            # self.format_data(idx=idx,label='Extracted SPO Qualifiers_scores (SPOQ)',df=self.spoq_scores,cols = ['spoq_pos_combo_score','s_token','p_token','o_token','pq_token','oq_token'])
            self.format_data(idx=idx,label='Semantic Roles Extracted from SPOQ',df=self.roles_scores,scores = ['spoq_score(j)'],cols=['program', 'action', 'client', 'direction', 'service'])
            self.format_data(idx=idx,label='No Roles Extracted from SPOQ',df=self.roles_other_scores,scores = ['spoq_score(j)'],cols=['subject', 'action', 'object', 'qualifier_predicate', 'qualifier_object'])
            self.format_data(idx=idx,label='SPOQ token scores',df=self.spoq_scores,scores = ['spoq_score(j)','s1','s2'],cols=['s_token','p_token','o_token','pq_token','oq_token'])
            # self.format_data(idx=idx,label='SPOQ token scores s1',df=self.spoq_scores,scores = ['s2'],cols=['s_token','p_token','o_token','pq_token','oq_token'])
            # self.format_data(idx=idx,label='SPOQ token scores s2',df=self.spoq_scores,scores = ['s1'],cols=['s_token','p_token','o_token','pq_token','oq_token'])


    def format_data(self,df,idx,label,scores,cols):
        tmp_df = df[df['idx'] == idx].copy()
        tmp_df = tmp_df.sort_values(by=scores,ascending=False)
        sentences = self.posp.sentences[self.posp.sentences.idx==idx].iloc[0]
        # sentences = sres.posp.sentences.loc[idx]
        texts = sent_tokenize(sentences['Text'])
        if len(self.idx_sentence_starts[idx]) != len(texts):
            self.results_errors.append([label,idx,"Sentence count for %s does not match POS roots, expected %s but found %s"%(idx,len(texts),len(self.idx_sentence_starts[idx]))])
            # return
        sent_starts = self.idx_sentence_starts[idx]
        sent_ends = self.idx_sentence_starts[idx][1:]
        sent_ends += [9999]
        sent_ranges = list(zip(sent_starts,sent_ends))

        for i,(start,end) in enumerate(sent_ranges):
            # print(start,end)
            if i not in self.results[idx].keys():
                self.results[idx][i] = {}
            self.results[idx][i]['text'] = texts[i]
            self.results[idx][i][label] = []
            self.results[idx][i][label].append("\t".join(scores+cols))
            found_any = False
            for _,row in tmp_df.iterrows():
                min_end = np.min([int(re.search(r"\-([0-9]+)$",t)[1]) for t in row[cols].values])
                if start <= min_end < end:
                    # print('   >>> ', min_end, row.values)
                    for c in scores:
                        row[c] = round(row[c],4)
                    tmp = "\t".join([str(r) for r in row[scores+cols].values])
                    self.results[idx][i][label].append(tmp)
                    found_any = True
                # else:
                #     # print('   >>> ', min_end, row.values)

            if not found_any:
                self.results[idx][i][label].append("none found")

    def to_file(self,filename='output.txt',ids=None):
        _ = os.makedirs(self.out_directory) if not os.path.exists(self.out_directory) else None
        res_content = ''
        if ids is None:
            results = self.results
        else:
            results = { k:self.results[k] for k in ids }
        for idx,parsed_dict in results.items():
            for sent_i,parsed in parsed_dict.items():
                res_content += "%s(%s): %s\n"%(idx,sent_i,parsed['text'])
                for key,result in parsed.items():
                    if key != 'text':
                        res_content += "\n%s\n"%key
                        res_content += "%s\n"%("\n".join(result))
                res_content += "\n"
            res_content += "\n"

        file = open(self.out_directory+filename, "w")
        _=file.write(res_content)
        file.close()

    def graph(self):
        data = [                ['Extracted Phrases Scores',sres.phrases_scores,['score(m)','text']],
                ['Extracted Corefs Scores',sres.corefs_scores, ['coref_score(i,t,cr)','ref','coref']],
                ['Extracted SPO token scores',sres.spo_scores, ['spo_score(i)','s_token','p_token','o_token']],
                ['Semantic Roles Extracted from SPOQ',sres.roles_scores, ['spoq_score(j)','program', 'action', 'client', 'direction', 'service']],
                ['No Roles Extracted from SPOQ',sres.roles_other_scores,['spoq_score(j)','subject', 'action', 'object', 'qualifier_predicate', 'qualifier_object']]
                # ['No Roles Extracted from SPOQ',sres.roles_other_scores,['spoq_score(j)','subject', 'action', 'object', 'qualifier_predicate', 'qualifier_object']]
        ]
        for label, df,cols in data:
            col = cols[0]
            plt.close()
            fig,ax = plt.subplots(1)
            ax.set_title(label)
            ax.set_xlabel(col)
            ax.set_ylabel('N')
            df[col].hist(ax=ax,density=True)
            fig.savefig('output/compass_full/reports/'+label+'.png')
        
