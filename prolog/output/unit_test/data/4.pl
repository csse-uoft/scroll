:- style_check(-discontiguous).
:- ensure_loaded("prolog/parsing").
:- dynamic coref/1.
text("The mental health services and laundry facilities, and women and men family services, giving clients a new start on life, are offered by The Lady Mother Program.").
gram(0,"ROOT",["ROOT-0", "NONE"],["offered-25", "VBN"]).
gram(1,"det",["services-4", "NNS"],["The-1", "DT"]).
gram(2,"amod",["health-3", "NN"],["mental-2", "JJ"]).
gram(3,"compound",["services-4", "NNS"],["health-3", "NN"]).
gram(4,"nsubj:pass",["offered-25", "VBN"],["services-4", "NNS"]).
gram(5,"cc",["facilities-7", "NNS"],["and-5", "CC"]).
gram(6,"amod",["facilities-7", "NNS"],["laundry-6", "JJ"]).
gram(7,"conj",["services-4", "NNS"],["facilities-7", "NNS"]).
gram(8,"punct",["services-4", "NNS"],[",-8", ","]).
gram(9,"cc",["women-10", "NNS"],["and-9", "CC"]).
gram(10,"conj",["services-4", "NNS"],["women-10", "NNS"]).
gram(11,"cc",["men-12", "NNS"],["and-11", "CC"]).
gram(12,"conj",["women-10", "NNS"],["men-12", "NNS"]).
gram(13,"compound",["services-14", "NNS"],["family-13", "NN"]).
gram(14,"dep",["women-10", "NNS"],["services-14", "NNS"]).
gram(15,"punct",["offered-25", "VBN"],[",-15", ","]).
gram(16,"advcl",["offered-25", "VBN"],["giving-16", "VBG"]).
gram(17,"iobj",["giving-16", "VBG"],["clients-17", "NNS"]).
gram(18,"det",["start-20", "NN"],["a-18", "DT"]).
gram(19,"amod",["start-20", "NN"],["new-19", "JJ"]).
gram(20,"obj",["giving-16", "VBG"],["start-20", "NN"]).
gram(21,"case",["life-22", "NN"],["on-21", "IN"]).
gram(22,"nmod",["start-20", "NN"],["life-22", "NN"]).
gram(23,"punct",["offered-25", "VBN"],[",-23", ","]).
gram(24,"aux:pass",["offered-25", "VBN"],["are-24", "VBP"]).
gram(25,"case",["Program-30", "NNP"],["by-26", "IN"]).
gram(26,"det",["Program-30", "NNP"],["The-27", "DT"]).
gram(27,"compound",["Program-30", "NNP"],["Lady-28", "NNP"]).
gram(28,"compound",["Program-30", "NNP"],["Mother-29", "NNP"]).
gram(29,"obl",["offered-25", "VBN"],["Program-30", "NNP"]).
gram(30,"punct",["offered-25", "VBN"],[".-31", "."]).
