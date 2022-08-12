from misc_lib import *
import re, json, unidecode, os, tqdm, glob
# from cairosvg import svg2png
from datetime import date
from nltk import Tree,ParentedTree
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.parse.stanford import StanfordDependencyParser, StanfordNeuralDependencyParser
# from stanfordcorenlp import CoreNLPDependencyParser
import nltk
# nltk.download('punkt')
from graphviz import Source
from stanfordcorenlp import StanfordCoreNLP
import openpyxl
# import spacy
# from spacy import displacy
# from corenlp_dtree_visualizer.converters import _corenlp_dep_tree_to_spacy_dep_tree
#import xlsxwriter

# for Windows installations without JAVAHOME set
# os.environ['JAVAHOME'] = "C:\Program Files (x86)\Common Files\Oracle\Java\javapath\java.exe"
class TreeParser:
    def __init__(self, posp=None, start_nlp=True, parser_detail='basic', ext='png'):
        self.posp = posp
        self.ext = ext
        if posp:
            self.directory = 'output/%s/tree/'%self.posp.file_prefix
            self.sentences = self.posp.sentences
        else:
            self.directory=None
            self.sentences = None
        self.parser_detail = parser_detail
        # CoreNLPClient
        if start_nlp:
            self.nlp = StanfordCoreNLP('http://localhost', 9000)
        else:
            self.nlp = None


        self.sdp = StanfordDependencyParser(
            path_to_jar = '../scroll/corenlp/stanford-parser-full-2020-11-17/stanford-parser.jar',
            path_to_models_jar = '../scroll/corenlp/stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar')

    def raw_parse(self,sentence):
        tp = self
        _ = os.makedirs(tp.directory) if not os.path.exists(tp.directory) else None

        ann = json.loads(
            unidecode.unidecode(
                tp.nlp.annotate(sentence,
                             properties={
                                #  'parse.model':'corenlp/stanford-parser-full-2020-11-17/stanford-parser.jar',
                                #  'depparse.model': "corenlp/stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar",
                                 'annotators': 'depparse', 'outputFormat': 'json', 'ner.useSUTime': 'false'})
            )
        )

        return tp.resolved_to_triples(ann)

    def resolved_to_triples(self,corenlp_output):
        tp = self
        if tp.parser_detail is None:
            tp.parser_detail = "basic"
        parsers = {
            "basic"       : "basicDependencies",
            "enhanced"    : "enhancedDependencies",
            "enhancedpp"  : "enhancedPlusPlusDependencies"}
        triples = []
        prev_i_max = -1
        i_offset = 0
        for sent in corenlp_output['sentences']:
            deps = sent[parsers[tp.parser_detail]]
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

    def get_dot(self,triples):
        re_com = re.compile(r'^(.+)\-([0-9]+)$')
        res = []
        dot_str = """
        digraph G{
        edge [dir=forward]
        node [shape=plaintext]
        """
        for edge in triples:
            s,p,o = edge
            sterm,snum = re_com.findall(s[0])[0]
            oterm,onum = re_com.findall(o[0])[0]
            dot_str += "%s [label=\"%s\n(%s)\"]\n"%(snum, sterm, s[1])
            dot_str += "%s -> %s [label=\"%s\"]\n"%(snum,onum,p)
            dot_str += "%s [label=\"%s\n(%s)\"]\n"%(onum, oterm, o[1])
        dot_str += "}\n"
        return dot_str

    def clean_sentence(self, text):
        text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', text)
        text = unidecode.unidecode(text)
        # make misc replacements
        text = text.strip().           \
                replace('as well as','and').\
                replace("&amp;", " and ").  \
                replace('%',' percent ').   \
                replace('+',' plus ').      \
                replace("\"","'").          \
                replace("``","'")
                
        # text = re.sub(r"[\-]+",' ', text)
        text = re.sub(r"[\*]+",'. ', text)
        text = re.sub(r"[\n\r]+",'. ', text)
        text = re.sub(r'[\-]+',' ', text)
        text = re.sub(r'[\.]+','. ', text)
        text = re.sub(r'[ ]+',' ', text)
        text = re.sub(r' \. ','. ', text)
        text = text.strip()
        return text


    def draw_sentences(self, view=False):
        tp = self
        i = 0
        for idx,text in tqdm.tqdm(self.sentences.values):
            text = re.sub(r'["“”]','',text.strip())
            if len(text) > 0 and text[0] != '#':
                outfile = "tree_%s_%s"%(idx,i)
                if tp.parser_detail == 'org':
                    tp.draw1(outfile, text, view=view, ext=tp.ext)
                else:
                    tp.draw(outfile, text, view=view, ext=tp.ext)
                i += 1

        work_files = glob.glob(tp.directory+'*')
        work_files =  [w for w in work_files if not w.endswith('.%s'%(tp.ext))]
        _=[os.remove(w) for w in work_files]


    def draw(self, filename, text, ext='png', view=False):
        tp = self
        _ = os.makedirs(tp.directory) if not os.path.exists(tp.directory) else None
        text = tp.clean_sentence(text)


        for i,sentence in enumerate(sent_tokenize(text)):
            out_filename = filename + '_'+str(i)
            results = tp.raw_parse(sentence)
            dep_tree_dot_repr = tp.get_dot(results)

            source = Source(dep_tree_dot_repr, format=ext)
            source.render(directory=tp.directory, filename=out_filename)
            os.remove(tp.directory+out_filename)
            if view:
                source.view()


    def draw1(self, filename, text, ext='png', view=False):
        tp = self
        _ = os.makedirs(tp.directory) if not os.path.exists(tp.directory) else None
        text = tp.clean_sentence(text)


        for i,sentence in enumerate(sent_tokenize(text)):
            out_filename = filename + '_'+str(i)
            try:
                resultold = list(tp.sdp.raw_parse(sentence))
                # _=[print(t) for t in list(resultold[0].triples())]
                dep_tree_dot_repr = [parse for parse in resultold][0].to_dot()
                source = Source(dep_tree_dot_repr, format=ext)
                source.render(directory=tp.directory, filename=out_filename)
                os.remove(tp.directory+out_filename)
                if view:
                    source.view()
            except (AssertionError, KeyError) as e:
                print('Error: '+out_filename)
                print(e)
                pass


