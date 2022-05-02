:- style_check(-discontiguous).
:- ensure_loaded("prolog/parsing").
:- dynamic coref/1.
text("The The Lady Mother Program offers mental health services.").
gram(0,"ROOT",["ROOT-0", "NONE"],["offers-6", "VBZ"]).
gram(1,"det",["Program-5", "NNP"],["The-1", "DT"]).
gram(2,"det",["Mother-4", "NNP"],["The-2", "DT"]).
gram(3,"compound",["Mother-4", "NNP"],["Lady-3", "NNP"]).
gram(4,"compound",["Program-5", "NNP"],["Mother-4", "NNP"]).
gram(5,"nsubj",["offers-6", "VBZ"],["Program-5", "NNP"]).
gram(6,"amod",["health-8", "NN"],["mental-7", "JJ"]).
gram(7,"compound",["services-9", "NNS"],["health-8", "NN"]).
gram(8,"obj",["offers-6", "VBZ"],["services-9", "NNS"]).
gram(9,"punct",["offers-6", "VBZ"],[".-10", "."]).
