:- style_check(-discontiguous).
:- ensure_loaded("prolog/parsing").
:- dynamic coref/1.
text("The Beach Hebrew Institute provided education services to Mary Johnson and John Smith, where she attended all night classes and he attended all day classes.").
gram(0,"ROOT",["ROOT-0", "NONE"],["provided-5", "VBD"]).
gram(1,"det",["Institute-4", "NNP"],["The-1", "DT"]).
gram(2,"compound",["Institute-4", "NNP"],["Beach-2", "NNP"]).
gram(3,"compound",["Institute-4", "NNP"],["Hebrew-3", "NNP"]).
gram(4,"nsubj",["provided-5", "VBD"],["Institute-4", "NNP"]).
gram(5,"compound",["services-7", "NNS"],["education-6", "NN"]).
gram(6,"obj",["provided-5", "VBD"],["services-7", "NNS"]).
gram(7,"case",["Johnson-10", "NNP"],["to-8", "IN"]).
gram(8,"compound",["Johnson-10", "NNP"],["Mary-9", "NNP"]).
gram(9,"obl",["provided-5", "VBD"],["Johnson-10", "NNP"]).
gram(10,"cc",["Smith-13", "NNP"],["and-11", "CC"]).
gram(11,"compound",["Smith-13", "NNP"],["John-12", "NNP"]).
gram(12,"conj",["Johnson-10", "NNP"],["Smith-13", "NNP"]).
gram(13,"punct",["provided-5", "VBD"],[",-14", ","]).
gram(14,"advmod",["attended-17", "VBD"],["where-15", "WRB"]).
gram(15,"nsubj",["attended-17", "VBD"],["she-16", "PRP"]).
gram(16,"advcl",["provided-5", "VBD"],["attended-17", "VBD"]).
gram(17,"det",["classes-20", "NNS"],["all-18", "DT"]).
gram(18,"compound",["classes-20", "NNS"],["night-19", "NN"]).
gram(19,"obj",["attended-17", "VBD"],["classes-20", "NNS"]).
gram(20,"cc",["attended-23", "VBD"],["and-21", "CC"]).
gram(21,"nsubj",["attended-23", "VBD"],["he-22", "PRP"]).
gram(22,"conj",["provided-5", "VBD"],["attended-23", "VBD"]).
gram(23,"det",["classes-26", "NNS"],["all-24", "DT"]).
gram(24,"compound",["classes-26", "NNS"],["day-25", "NN"]).
gram(25,"obj",["attended-23", "VBD"],["classes-26", "NNS"]).
gram(26,"punct",["provided-5", "VBD"],[".-27", "."]).
coref([["Johnson-10", "NNP"],["she-16", "PRP"]]).
coref([["Mary-9", "NNP"],["she-16", "PRP"]]).
coref([["John-12", "NNP"],["he-22", "PRP"]]).
coref([["Smith-13", "NNP"],["he-22", "PRP"]]).