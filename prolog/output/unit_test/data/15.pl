:- style_check(-discontiguous).
:- ensure_loaded("prolog/parsing").
:- dynamic coref/1.
text("The mental health services are offered by The Lady Mother Program.").
gram(0,"ROOT",["ROOT-0", "NONE"],["offered-6", "VBN"]).
gram(1,"det",["services-4", "NNS"],["The-1", "DT"]).
gram(2,"amod",["health-3", "NN"],["mental-2", "JJ"]).
gram(3,"compound",["services-4", "NNS"],["health-3", "NN"]).
gram(4,"nsubj:pass",["offered-6", "VBN"],["services-4", "NNS"]).
gram(5,"aux:pass",["offered-6", "VBN"],["are-5", "VBP"]).
gram(6,"case",["Program-11", "NNP"],["by-7", "IN"]).
gram(7,"det",["Program-11", "NNP"],["The-8", "DT"]).
gram(8,"compound",["Program-11", "NNP"],["Lady-9", "NNP"]).
gram(9,"compound",["Program-11", "NNP"],["Mother-10", "NNP"]).
gram(10,"obl",["offered-6", "VBN"],["Program-11", "NNP"]).
gram(11,"punct",["offered-6", "VBN"],[".-12", "."]).
