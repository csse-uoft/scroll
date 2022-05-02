:- style_check(-discontiguous).
:- ensure_loaded("prolog/parsing").
:- dynamic coref/1.
text("The annual city market offers fresh produce for local families.").
gram(0,"ROOT",["ROOT-0", "NONE"],["offers-5", "VBZ"]).
gram(1,"det",["market-4", "NN"],["The-1", "DT"]).
gram(2,"amod",["market-4", "NN"],["annual-2", "JJ"]).
gram(3,"compound",["market-4", "NN"],["city-3", "NN"]).
gram(4,"nsubj",["offers-5", "VBZ"],["market-4", "NN"]).
gram(5,"amod",["produce-7", "NN"],["fresh-6", "JJ"]).
gram(6,"obj",["offers-5", "VBZ"],["produce-7", "NN"]).
gram(7,"case",["families-10", "NNS"],["for-8", "IN"]).
gram(8,"amod",["families-10", "NNS"],["local-9", "JJ"]).
gram(9,"nmod",["produce-7", "NN"],["families-10", "NNS"]).
gram(10,"punct",["offers-5", "VBZ"],[".-11", "."]).
