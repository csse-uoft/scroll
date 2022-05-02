:- style_check(-discontiguous).
:- ensure_loaded("prolog/parsing").
:- dynamic coref/1.
text("The annual city market offers local families with fresh produce.").
gram(0,"ROOT",["ROOT-0", "NONE"],["offers-5", "VBZ"]).
gram(1,"det",["market-4", "NN"],["The-1", "DT"]).
gram(2,"amod",["market-4", "NN"],["annual-2", "JJ"]).
gram(3,"compound",["market-4", "NN"],["city-3", "NN"]).
gram(4,"nsubj",["offers-5", "VBZ"],["market-4", "NN"]).
gram(5,"amod",["families-7", "NNS"],["local-6", "JJ"]).
gram(6,"obj",["offers-5", "VBZ"],["families-7", "NNS"]).
gram(7,"case",["produce-10", "NN"],["with-8", "IN"]).
gram(8,"amod",["produce-10", "NN"],["fresh-9", "JJ"]).
gram(9,"obl",["offers-5", "VBZ"],["produce-10", "NN"]).
gram(10,"punct",["offers-5", "VBZ"],[".-11", "."]).
