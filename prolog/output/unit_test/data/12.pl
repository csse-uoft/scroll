:- style_check(-discontiguous).
:- ensure_loaded("prolog/parsing").
:- dynamic coref/1.
text("Alberta Health provides annual financial coverage for a standard eye exam for all Albertans age 65 and older.").
gram(0,"ROOT",["ROOT-0", "NONE"],["provides-3", "VBZ"]).
gram(1,"compound",["Health-2", "NNP"],["Alberta-1", "NNP"]).
gram(2,"nsubj",["provides-3", "VBZ"],["Health-2", "NNP"]).
gram(3,"amod",["coverage-6", "NN"],["annual-4", "JJ"]).
gram(4,"amod",["coverage-6", "NN"],["financial-5", "JJ"]).
gram(5,"obj",["provides-3", "VBZ"],["coverage-6", "NN"]).
gram(6,"case",["exam-11", "NN"],["for-7", "IN"]).
gram(7,"det",["exam-11", "NN"],["a-8", "DT"]).
gram(8,"amod",["exam-11", "NN"],["standard-9", "JJ"]).
gram(9,"compound",["exam-11", "NN"],["eye-10", "NN"]).
gram(10,"nmod",["coverage-6", "NN"],["exam-11", "NN"]).
gram(11,"case",["Albertans-14", "NNPS"],["for-12", "IN"]).
gram(12,"det",["Albertans-14", "NNPS"],["all-13", "DT"]).
gram(13,"nmod",["exam-11", "NN"],["Albertans-14", "NNPS"]).
gram(14,"xcomp",["provides-3", "VBZ"],["age-15", "NN"]).
gram(15,"nummod",["age-15", "NN"],["65-16", "CD"]).
gram(16,"cc",["older-18", "JJR"],["and-17", "CC"]).
gram(17,"conj",["age-15", "NN"],["older-18", "JJR"]).
gram(18,"punct",["provides-3", "VBZ"],[".-19", "."]).
