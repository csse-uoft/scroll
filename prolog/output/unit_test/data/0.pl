:- style_check(-discontiguous).
:- ensure_loaded("prolog/parsing").
:- dynamic coref/1.
text("Bob Smith and Mary Johnson, farm animals and poultry farmers went to the grocery store, downtown summer park, then drove to the picnic table, urban hospital, and shopping centre.").
gram(0,"ROOT",["ROOT-0", "NONE"],["went-12", "VBD"]).
gram(1,"compound",["Smith-2", "NNP"],["Bob-1", "NNP"]).
gram(2,"nsubj",["went-12", "VBD"],["Smith-2", "NNP"]).
gram(3,"cc",["Johnson-5", "NNP"],["and-3", "CC"]).
gram(4,"compound",["Johnson-5", "NNP"],["Mary-4", "NNP"]).
gram(5,"conj",["Smith-2", "NNP"],["Johnson-5", "NNP"]).
gram(6,"punct",["Johnson-5", "NNP"],[",-6", ","]).
gram(7,"compound",["animals-8", "NNS"],["farm-7", "NN"]).
gram(8,"conj",["Johnson-5", "NNP"],["animals-8", "NNS"]).
gram(9,"cc",["farmers-11", "NNS"],["and-9", "CC"]).
gram(10,"compound",["farmers-11", "NNS"],["poultry-10", "NN"]).
gram(11,"conj",["Johnson-5", "NNP"],["farmers-11", "NNS"]).
gram(12,"case",["park-20", "NN"],["to-13", "IN"]).
gram(13,"det",["park-20", "NN"],["the-14", "DT"]).
gram(14,"compound",["store-16", "NN"],["grocery-15", "NN"]).
gram(15,"compound",["park-20", "NN"],["store-16", "NN"]).
gram(16,"punct",["park-20", "NN"],[",-17", ","]).
gram(17,"dep",["park-20", "NN"],["downtown-18", "NN"]).
gram(18,"compound",["park-20", "NN"],["summer-19", "NN"]).
gram(19,"obl",["went-12", "VBD"],["park-20", "NN"]).
gram(20,"punct",["went-12", "VBD"],[",-21", ","]).
gram(21,"advmod",["drove-23", "VBD"],["then-22", "RB"]).
gram(22,"dep",["went-12", "VBD"],["drove-23", "VBD"]).
gram(23,"case",["table-27", "NN"],["to-24", "IN"]).
gram(24,"det",["table-27", "NN"],["the-25", "DT"]).
gram(25,"compound",["table-27", "NN"],["picnic-26", "NN"]).
gram(26,"obl",["drove-23", "VBD"],["table-27", "NN"]).
gram(27,"punct",["table-27", "NN"],[",-28", ","]).
gram(28,"amod",["hospital-30", "NN"],["urban-29", "JJ"]).
gram(29,"conj",["table-27", "NN"],["hospital-30", "NN"]).
gram(30,"punct",["table-27", "NN"],[",-31", ","]).
gram(31,"cc",["centre-34", "NN"],["and-32", "CC"]).
gram(32,"compound",["centre-34", "NN"],["shopping-33", "NN"]).
gram(33,"conj",["table-27", "NN"],["centre-34", "NN"]).
gram(34,"punct",["went-12", "VBD"],[".-35", "."]).