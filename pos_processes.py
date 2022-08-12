from misc_lib import *
import numpy as np
import tqdm, os
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


class POSProcesses:
    def __init__(self, col='Description', program_name='Name', dep_parser='basic'):
        _ = os.makedirs('output') if not os.path.exists('output') else None
        self.dep_parser = dep_parser
        self.deps = []
        self.sentences = None

        self.program_names = None
        # self.cols = ['Name','Description','SearchWords','HoursOfOperation','HowYouDefineYourOrganization','PostalCode','Eligibility','AttachedPrograms']
        self.col =  col
        self.program_name = program_name

        self.indexed_sentences = None
        # self.dependency_parser = StanfordDependencyParser(
        #     path_to_jar = 'output/stanford-parser-full-2020-11-17/stanford-parser.jar',
        #     path_to_models_jar = 'output/stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar')


    # obsolete: used with StanfordDependencyParser()
    #           replaced by StanfordCoreNLP()
    # def deps_to_triples(self,deps):
    #     triples = []
    #     for t1 in deps.nodes.values():
    #         label1 = "%s-%s"%(t1['word'],t1['address'])
    #         for dep,t2is in t1['deps'].items():
    #             for t2i in t2is:
    #                 t2 = deps.nodes[t2i]
    #                 label2 = "%s-%s"%(t2['word'],t2['address'])
    #                 triples.append([[label1,t1['tag']],dep,[label2,t2['tag']]])
    #     return triples

    def generate_sentence_with_term_indexes(self, triples):
        re_compiled = re.compile(r'(.*)\-([0-9]+)')
        def assign_term(res,term):
            if term[1] != 'NONE':
                match = re.search(re_compiled,term[0])
                if match:
                    res[int(match[2])] = match[0]
            return res
        res = [None]*len(triples)*2
        for _,[s,p,o] in triples:
            assign_term(res,s)
            # assign_term(res,p)
            assign_term(res,o)
        return '#DEL#'.join([s for s in res if s is not None])

    def resolved_to_triples(self,corenlp_output):
        parser_detail = {
            "basic"       : "basicDependencies",
            "enhanced"    : "enhancedDependencies",
            "enhancedpp"  : "enhancedPlusPlusDependencies"}[self.dep_parser]


        triples = []
        prev_i_max = -1
        i_offset = 0
        for sent in corenlp_output['sentences']:
            deps = sent[parser_detail]
            tokens = sent['tokens']
            for v in deps:
                w1 = v['governorGloss']
                w2 = v['dependentGloss']
                i1 = v['governor'] + i_offset
                i2 = v['dependent'] + i_offset
                prev_i_max = np.max([prev_i_max,i1,i2])
                pos1 = next((x['pos'] for x in tokens if x['index'] == (i1-i_offset)), 'NONE')
                pos2 = next((x['pos'] for x in tokens if x['index'] == (i2-i_offset)), 'NONE')
                triples.append([["%s-%s"%(w1,i1),pos1], v['dep'], ["%s-%s"%(w2,i2),pos2]])
            i_offset = prev_i_max + 1
        return triples

    def set_tests(self):
        self.test_configs = [
                {'file' : "output/%s/res/spo_results.txt",
                'cmd'  : "findall([W,D],(spo(D,W)),L),remove_duplicates(L,L2),L2\=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/spo_noresults.txt",
                'cmd'  : "findall([W,D],(spo(D,W)),L),remove_duplicates(L,L2),L2=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/phrase_results.txt",
                'cmd'  : "findall([W,D],(phrase_([_,D],W)),L),remove_duplicates(L,L2),L2\=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/phrase_noresults.txt",
                'cmd'  : "findall([W,D],(phrase_([_,D],W)),L),remove_duplicates(L,L2),L2=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/phrase_pos_results.txt",
                'cmd'  : "findall([W,D],(phrase_pos([_,D],W)),L),remove_duplicates(L,L2),L2\=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/phrase_pos_noresults.txt",
                'cmd'  : "findall([W,D],(phrase_pos([_,D],W)),L),remove_duplicates(L,L2),L2=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/mod_results.txt",
                'cmd'  : "findall([W,D],(mod_(D,W)),L),remove_duplicates(L,L2),L2\=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/mod_noresults.txt",
                'cmd'  : "findall([W,D],(mod_(D,W)),L),remove_duplicates(L,L2),L2=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/coref_results.txt",
                'cmd'  : "findall([[0],D],(coref(D)),L),remove_duplicates(L,L2),L2\=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/coref_noresults.txt",
                'cmd'  : "findall([[0],D],(coref(D)),L),remove_duplicates(L,L2),L2=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/conj_results.txt",
                'cmd'  : "findall([[0],D],(conj(D)),L),remove_duplicates(L,L2),L2\=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/conj_noresults.txt",
                'cmd'  : "findall([[0],D],(conj(D)),L),remove_duplicates(L,L2),L2=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/num_results.txt",
                'cmd'  : "findall([[0],D,NUM],(num(D,NUM)),L),remove_duplicates(L,L2),L2\=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
              
                {'file' : "output/%s/res/focus_results.txt",
                'cmd'  : "findall([W,D],(focus(D,W)),L),remove_duplicates(L,L2),L2\=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/focus_noresults.txt",
                'cmd'  : "findall([W,D],(focus(D,W)),L),remove_duplicates(L,L2),L2=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/spo_focus_results.txt",
                'cmd'  : "findall([W,[[F,FPOS],P,O]],(focus([[F,FPOS],[FPN]],_),spo([[F,FPOS],P,O],W)),L),remove_duplicates(L,L2),L2\=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/spo_focus_noresults.txt",
                'cmd'  : "findall([W,[[F,FPOS],P,O]],(focus(F,_),spo([[F,FPOS],P,O],W)),L),remove_duplicates(L,L2),L2=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/spo_focus_spo_qualifier_results.txt",
                'cmd'  : "findall([W,F],(focus(F,_),spo_qualifier(F,W)),L),remove_duplicates(L,L2),L2\=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/spo_qualifier_results.txt",
                'cmd'  : "findall([W,D],(spo_qualifier(D,W)),L),remove_duplicates(L,L2),L2\=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/spo_qualifier_noresults.txt",
                'cmd'  : "findall([W,D],(spo_qualifier(D,W)),L),remove_duplicates(L,L2),L2=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/program_to_service_results.txt",
                'cmd'  : "findall([W,D],(program_offers_service(D,W)),L),remove_duplicates(L,L2),L2\=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
                {'file' : "output/%s/res/program_to_service_noresults.txt",
                'cmd'  : "findall([W,D],(program_offers_service(D,W)),L),remove_duplicates(L,L2),L2=[],text(Text),writeln([%s,Text]),findall(_,(member(T,L2),writeln([%s,T])),_)",
                'ids_n'  : 2},
            ]
        return self.test_configs

    def run_tests(self, test_ids=None):
        directory = 'output/'+self.file_prefix+'/res/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        tests = self.set_tests()
        if test_ids is not None and len(test_ids)>0:
            tests = (np.array(self.test_configs)[test_ids]).tolist()

        for test_config in tests:
            service_file = test_config['file']%(self.file_prefix)   
            search_cmd = test_config['cmd']
            ids_n = test_config['ids_n']
            _=os.system("rm -f %s" %(service_file))
            _=os.system("touch %s"%(service_file))
            for filename in tqdm.tqdm(sorted(glob.glob('output/%s/data/*.pl'%(self.file_prefix)))):
                # print(filename)
                try:
                    i =re.search(r'([0-9]+)\.pl',filename)[1]
                except TypeError:
                    continue
                cmd = "swipl -s %s -g \""%(filename) + search_cmd%tuple([i]*ids_n) + ".\" -g halt >> %s"%(service_file)
                _=os.system(cmd)


    def resolve_coref(self,corenlp_output):
        """ Transfer the word form of the antecedent to its associated pronominal anaphor(s) """
        for coref in corenlp_output['corefs']:
            mentions = corenlp_output['corefs'][coref]
            antecedent = mentions[0]  # the antecedent is the first mention in the coreference chain
            for j in range(1, len(mentions)):
                mention = mentions[j]
                if mention['type'] == 'PRONOMINAL':
                    # get the attributes of the target mention in the corresponding sentence
                    target_sentence = mention['sentNum']
                    target_token = mention['startIndex'] - 1
                    # transfer the antecedent's word form to the appropriate token in the sentence
                    corenlp_output['sentences'][target_sentence - 1]['tokens'][target_token]['word'] = antecedent['text']


    def combine_resolved_coref(self,corenlp_output):
        """ Print the "resolved" output """
        res = []
        possessives = ['hers', 'his', 'their', 'theirs','he','her',]
        for sentence in corenlp_output['sentences']:
            for ti,token in enumerate(sentence['tokens']):
                output_word = token['word']
                output_orgword = token['originalText']
                coref = 1 if token['word'] != token['originalText'] else 0
                regexPattern = '|'.join(map(re.escape, [',',' ',';','and','or']))
                sub_tokens = [t for t in re.split(regexPattern,token['word']) if len(t) > 0]
                    
                # check lemmas as well as tags for possessive pronouns in case of tagging errors
                if token['lemma'] in possessives or token['pos'] == 'PRP$':
                    output_word += "'s"  # add the possessive morpheme
                output_word += token['after']
                output_orgword += token['after']
                res.append([ti+1,output_word, coref, output_orgword, token['originalText'], sub_tokens])
        return res

    # build triples with resolved coreferences
    def resolve_triples(self,triples, res,which='left'):
        corefs = copy.deepcopy([c for c in res if c[2] == 1])
        out_triples = []
        for coref in copy.deepcopy(corefs):

            tokens = set(coref[5])
            ii,jj = (0,2) if which =='left' else (2,0)
            # change subject in s-p-o
            to_change = copy.deepcopy([t for t in triples if t[1][ii][0].split('-')[0] == coref[4]])
            for new_trp1,token in itertools.product(to_change,tokens): 
                new_trp = copy.deepcopy(new_trp1)
                from_change = [t[1][ii] for t in triples if t[1][ii][0].split('-')[0] == token]
                from_change += [t[1][jj] for t in triples if t[1][jj][0].split('-')[0] == token]
                for from_token in unique_all(from_change):
                    new_trp[1][ii] = from_token
                    out_triples.append(new_trp)

        return unique_all(out_triples)
        
    def replace_resolved_triples(self,triples, replaced):
        # replace triple corefs with newly built ones
        to_replace = set([t[0] for t in replaced])
        replaced_corefs = []
        out_triples = []
        for t1 in triples:
            if t1[0] not in to_replace:
                out_triples.append(t1)
            else:
                t2 = [tt1 for tt1 in replaced if t1[0] == tt1[0]]
                out_triples += t2
                replaced_corefs += [(tt,t1[1][2]) for _,[_,_,tt] in t2]

        return {'triples':out_triples, 'replaced':replaced_corefs}

    def clean_sentences(self):
        res_directory = 'output/'+self.file_prefix+'/res/'
        for idx,sentence in self.sentences.iterrows():
            # insert space between sentences if one does not exist
            text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', sentence['Text'])
            # make misc replacements
            text = text.strip().                \
                    replace('as well as','and').\
                    replace("&amp;", " and ").  \
                    replace('-',' ').           \
                    replace('%',' percent ').   \
                    replace('+',' plus ').      \
                    replace("\"","'").          \
                    replace("\n"," ").          \
                    replace("\r"," ").          \
                    strip()
            self.sentences.at[idx,'Text'] = text
        self.sentences.to_csv(res_directory+'sentences.csv',index=True)

    def build_prolog_files(self):
        self.indexed_sentences = pd.DataFrame(columns=['idx','indexed_text'])
        directory = 'output/'+self.file_prefix+'/data/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # path_to_jar = 'output/stanford-parser-full-2020-11-17/stanford-parser.jar'
        # path_to_models_jar = 'output/stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar'

        # start coref server
        nlp = StanfordCoreNLP('http://localhost',9000)
        # step 1: clean data
        self.clean_sentences()

        # get data
        # step 2: parse data
        self.build_errors = []
        for i,text in tqdm.tqdm(self.sentences.values, total=len(self.sentences)):
            # 1) preprocess text
            try:
                # 2) resolve coreferences
                output = json.loads(
                    unidecode.unidecode(
                        nlp.annotate(text, properties= {'annotators':'dcoref','outputFormat':'json','ner.useSUTime':'false'})
                    )
                )
            except:
                self.build_errors.append(i)
                continue

            self.resolve_coref(output)
            res = self.combine_resolved_coref(output)
            # _=[print(r) for r in res if r[2] == 1]
            # text2 = ''.join([r[1] for r in res])
            # text_corefs = ''.join([r[3] for r in res])

            # generate triples
            triples =list(enumerate(self.resolved_to_triples(output)))

            # create list with work index
            indexed_text = self.generate_sentence_with_term_indexes(triples)
            # Note: appending directly into the DataFrame to ensure already processed records are stored
            self.indexed_sentences = self.indexed_sentences.append(pd.Series([i,indexed_text], index=['idx','indexed_text']), ignore_index=True)

            # 3) build triples using coref
            triples1 = self.resolve_triples(triples,res,which='left')
            tmp1 = self.replace_resolved_triples(triples,triples1)
            triples_1 = tmp1['triples']
            # replaced_triples_1 = tmp1['replaced']

            triples2 = self.resolve_triples(triples_1,res,which='right')
            tmp2 = self.replace_resolved_triples(triples_1,triples2)
            new_triples = tmp2['triples']
            replaced_triples = tmp2['replaced']


            # build  pl file content
            res_content =  ":- style_check(-discontiguous).\n"
            res_content += ":- ensure_loaded(\"prolog/parsing\").\n"
            res_content += ":- dynamic coref/1.\n"
            res_content += 'text("%s").\n'%(str(text))
            res_content += ("\n".join(["gram(%s,\"%s\",%s,%s)."%(i,t,[l1,l2],[r1,r2]) for i,[[l1,l2],t,[r1,r2]] in triples])).replace("['","[\"").replace("']","\"]").replace("',","\",").replace(", '", ", \"")
            res_content += "\n"
            res_content += ("\n".join(["coref([%s,%s])."%([l1,l2],[r1,r2]) for ([l1,l2],[r1,r2]) in replaced_triples]).replace("['","[\"").replace("']","\"]").replace("',","\",").replace(", '", ", \""))

            fl = str(i)
            filename = "output/%s/data/%s.pl"%(self.file_prefix,fl)
            file = open(filename, "w")
            _=file.write(res_content)
            file.close()



    def load_test_sentences(self,filename='output/test_sentences.txt', file_prefix = "test"):
        self.file_prefix = file_prefix
        directory = 'output/'+file_prefix
        if not os.path.exists(directory):
            os.makedirs(directory)
        res_directory = 'output/'+file_prefix+'/res/'
        if not os.path.exists(res_directory):
            os.makedirs(res_directory)
        
        with open(filename) as f:
            self.sentences = f.readlines()
        self.sentences = [s.strip() for s in self.sentences if len(s.strip()) > 0 and s.strip()[0] != '#']
        self.sentences = pd.DataFrame(list(enumerate(self.sentences))[:min(100, len(self.sentences))], columns=['idx','Text'])

        self.sentences.to_csv(res_directory+'sentences.csv',index=False)
        # file[['Name']].loc[idx].to_csv(res_directory+'program_names.csv', index=True)


    def load_compass_sentences(self,filename='models/sent_pos_full_Alberta.pkl',random_n=10, file_prefix = "compass"):
        self.file_prefix = file_prefix
        directory = 'output/'+file_prefix
        if not os.path.exists(directory):
            os.makedirs(directory)
        res_directory = 'output/'+file_prefix+'/res/'
        if not os.path.exists(res_directory):
            os.makedirs(res_directory)

        filehandler = open(filename, 'rb') 
        sent_pos = pickle.load(filehandler)
        np.random.seed(42)
        predicates = ['provides','offer','provide','provided','offered']

        sentences = [(i,sent) for i,sent in enumerate(sent_pos.processed) if sent_pos.roots[i][0] in predicates]
        sentences = np.array(sentences)
        idx = np.random.choice(len(sentences), random_n, replace=False)
        idx.sort()

        self.sentences = sentences[idx]
        self.sentences.columns =['idx','Text']
        sentences.to_csv(res_directory+'sentences.csv',index=True)
        # file[['Name']].loc[idx].to_csv(res_directory+'program_names.csv', index=True)


    def load_compass_paragraphs(self, filename='data/fwdontologynextsteps/Alberta full data set.csv', random_n=10,file_prefix='compass_para'):
        self.file_prefix = file_prefix
        directory = 'output/'+file_prefix
        if not os.path.exists(directory):
            os.makedirs(directory)
        res_directory = 'output/'+file_prefix+'/res/'
        if not os.path.exists(res_directory):
            os.makedirs(res_directory)

        file = pd.read_csv(filename)
        df = file[self.col]

        np.random.seed(42)
        predicates = ['provide','offer']
        sentences = df[df.str.contains('|'.join(predicates))]

        if random_n is not None:
            sentences = sentences.sample(random_n).sort_index()

        idx = sentences.index.tolist()
        idx.sort()

        self.sentences = sentences.reset_index(drop=False)
        self.program_names = file[self.program_name].loc[idx]
        self.sentences.columns =['idx','Text']
        self.sentences.to_csv(res_directory+'sentences.csv',index=True)
        file[[self.program_name]].loc[idx].to_csv(res_directory+'program_names.csv', index=True)

    def prolog_to_csv(self):
        parse_re = re.compile('([^\[,]+,)|(\[[^\]]+\])|([\]],[^,]+,)|([\]],[^,]+)|([^\],]+)$|(^[^\[,]+)')
        for filename in tqdm.tqdm(glob.glob('output/%s/res/*results.txt'%(self.file_prefix))):
            with open(filename) as f:
                rows = f.readlines()
            res_content = ''
            prev_idx = ''
            for row in rows:
                matches = re.search(r"\[([0-9]+),(.*)\]",row.strip())
                idx = matches[1]
                data = matches[2]
                if idx != prev_idx:
                    # assuming first record is the sentence itself
                    res_content += "%s\t\"TEXT\"\t\"%s\"\n"%(idx,data)
                else:
                    # parsing componentns that may be single items, or list of itmes grouped by "[x]" or "(x)".
                    # all seperated by commas ","
                    data = data[1:-1]
                    fields = [[ rr.strip('[](),') for rr in r if rr!=''][0] for r in re.findall(parse_re,data)]
                    res_content += idx+"\t\""+("\"\t\"".join(fields)) + "\"\n"
                prev_idx = idx

            filename2 =  filename.replace('.txt','_format.txt')
            file = open(filename2, "w")
            _=file.write(res_content)
            file.close()
    def gen_summaries(self):
        print()
        # gen POS tree

        # gen POS tags

        # gen phrases, scored
        
        # gen focus

        # gen coref

        # gen SPOs

        # gen SPOQs

        # gen proram, service, client

