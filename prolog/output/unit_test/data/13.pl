:- style_check(-discontiguous).
:- ensure_loaded("prolog/parsing").
:- dynamic coref/1.
text("The Legion also provides representation to Veterans and their families.").
gram(0,"ROOT",["ROOT-0", "NONE"],["provides-4", "VBZ"]).
gram(1,"det",["Legion-2", "NNP"],["The-1", "DT"]).
gram(2,"nsubj",["provides-4", "VBZ"],["Legion-2", "NNP"]).
gram(3,"advmod",["provides-4", "VBZ"],["also-3", "RB"]).
gram(4,"obj",["provides-4", "VBZ"],["representation-5", "NN"]).
gram(5,"case",["Veterans-7", "NNP"],["to-6", "IN"]).
gram(6,"obl",["provides-4", "VBZ"],["Veterans-7", "NNP"]).
gram(7,"cc",["families-10", "NNS"],["and-8", "CC"]).
gram(8,"nmod:poss",["families-10", "NNS"],["their-9", "PRP$"]).
gram(9,"conj",["Veterans-7", "NNP"],["families-10", "NNS"]).
gram(10,"punct",["provides-4", "VBZ"],[".-11", "."]).
