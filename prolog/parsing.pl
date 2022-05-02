:- style_check(-discontiguous).
:- ensure_loaded(program_service_types).

remove_duplicates([], []).

remove_duplicates([Head | Tail], Result) :-
    member(Head, Tail), !,
    remove_duplicates(Tail, Result).

remove_duplicates([Head | Tail], [Head | Result]) :-
    remove_duplicates(Tail, Result).

incr_list(L) :-
	L=[L1|_],last(L,L2),findall(I,between(L1,L2,I),L).

flatten2([], []) :- !.
flatten2([L|Ls], FlatL) :-
    !,
    flatten2(L, NewL),
    flatten2(Ls, NewLs),
    append(NewL, NewLs, FlatL).
flatten2(L, [L]).
last(L,X) :- length(L,Len),nth1(Len,L,X).
first([X|_],X).

diff(L1,L2,DIFF) :- 
	subtract(L1,L2,R1),subtract(L2,L1,R2), 
	append(R1,R2,R),remove_duplicates(R,DIFF).


term_num(T,Num) :-
	split_string(T,"-","",Terms),last(Terms,NumStr),
	atom_number(NumStr,Num).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

conj([[X,XPOS],[Y,YPOS]]) :-
	gram(_,"conj",[X,XPOS],[Y,YPOS]),
	gram(_,_,[Y,YPOS],_).
conj([[X,XPOS],[Y,YPOS]]) :-
	gram(_,"conj",[X,XPOS],[Y,YPOS]).
conj([[X,XPOS],[Y,YPOS]]) :-
	gram(_,"dep",[X,XPOS],[Y,YPOS]),
	verb_tag(XPOS),verb_tag(YPOS).
% dep is a fallback relation if one is not infered properly.
conj([[Z,ZPOS],[X,XPOS]]) :-
	gram(_,"conj",[X,XPOS],_),
	gram(_,"dep",[X,XPOS],[Z,ZPOS]).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% top level SPO
spo(R,W) :-spo_2(R,W).
%% spo(R,W) :-spo_3(R,W).
% level 2 SPO
%% spo_2old(R,[10|W]) :-spo_1(R,W).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Level 2 SPOs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Level 1 SPOs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  simple S provides O.
%% exludes S provides O1 with O, where O is the provided term.
spo_1([S,P,O], [1]) :-
	gram(_,"nsubj", P,S),
	gram(_,"obj", P,O),
	not(gram(_,"obl", P,_)).

% S [went] to O where went=P
spo_1([S,P,O], [1]) :-
	gram(_,"nsubj", P,S),
	gram(_,"obl", P,O),
	not(gram(_,"obj", P,_)).
%%  S [provides] O to X, where the receiving end is the O
spo_1([S,P,O], [2]) :-
	gram(_,"nsubj", P,S),
	gram(_,"obj",P,O),
	gram(_,"obl",P,O1),
	gram(_,"case",O1,P1),
	[PP,_]=P1,
	split_string(PP,"-","",[PTerm|_]),
	member(PTerm,["to","for"]).
%% S provides O2 with O, where O is the target object
spo_1([S,P,O], [2]) :-
	gram(_,"nsubj", P,S),
	gram(_,"obj", P,_),
	gram(_,"obl", P,O),
	gram(_,"case",O,P1),
	[PTerm,_]=P1,
	split_string(PTerm,"-","",[P1POS|_]),
	P1POS="with".
spo_1([P,P1,O], 2) :-
	gram(_,"nsubj", P,_),
	gram(_,"obj", P,_),
	gram(_,"obl", P,O),
	gram(_,"case",O,P1),
	[PTerm,_]=P1,
	split_string(PTerm,"-","",[P1POS|_]),
	P1POS="on".
%% X provides S with O, where S is the target subject
spo_1([O,P1,O1], [2]) :-
	gram(_,"nsubj", P,_),
	gram(_,"obj", P,O),
	gram(_,"obl", P,O1),
	gram(_,"case",O1,P1),
	[PTerm,_]=P1,
	split_string(PTerm,"-","",[P1POS|_]),
	P1POS="with".

spo_1([S,P,O], [1]) :-
	gram(_,"nsubj", P,S),
	gram(_,"advcl", P,_),
	gram(_,"obl", P,O).

spo_1([S,P,O], [1]) :-
	gram(_,"nsubj", P1,S),
	gram(_,"xcomp", P1,P),
	gram(_,"obj", P,O).

spo_1([S,P,O], [1]) :-
	gram(_,"nsubj:pass", P,O),
	gram(_,"obl", P,S).
spo_1([S,P,O], [1]) :-
	gram(_,"nsubj:pass", _,S),
	gram(_,"acl", S,P),
	gram(_,"obj", P,O).
spo_1([S,P,O], [3]) :-
	gram(_,"acl", O,P),
	(_,PPOS)=P,
	verb_tag(PPOS),
	gram(_,"obl", P,S).

spo_1([S,P,O], [3]) :-
	gram(_,"acl", S,P),
	gram(_,"obj", P,O).
%%  X provides S to O, where the receiving end is the 
spo_1([S,P,O], [2]) :-
	gram(_,"nsubj", P1,_),
	gram(_,"obj",P1,S),
	gram(_,"obl",P1,O),
	gram(_,"case",O,P),
	[PP,_]=P,
	split_string(PP,"-","",[PTerm|_]),
	member(PTerm,["to","for"]).
% a possessive pronoun implies verb="has"
spo_1([S,("has","VBZ"),O], [4]) :-
	gram(_,"amod", O,S),
	(_,"PRP$")=S.
spo_1([S,("has","VBZ"),O], [4]) :-
	gram(_,"nmod:poss", O,S),
	(_,"PRP$")=S.

spo_1([S,P,O], [5]) :-
	gram(_,"nsubj",P1,S),
	gram(_,"acl:relcl",O,P1),
	gram(_,"xcomp",P1,P).
spo_1([S,P,O], [5]) :-
	gram(_,"acl:relcl",S,P),
	gram(_,"ccomp",P,O).

%% spo_1([S,P,O]) :-
%% 	gram(_,"ccomp",S,P),
%% 	gram(_,"nsubj",P,O).
spo_1([S,P,O], [5]) :-
	gram(_,"ccomp",S,P),
	gram(_,"obj",P,O).
% 1 - dirrect
% 2 - with case: to, from, with
% 3 - acl with verb
% 4 - PRP with inserted predicate
% 5 - with ccomp/xcomp





% qualifier
% connects two SPOs or 
spo_qualifier([S,P,O,PQ,OQ], [21]) :-
	gram(_,"nsubj", P,S),
	gram(_,"obj", P,O),
	gram(_,"obl", P,OQ),
	gram(_,"case",OQ,PQ),
	O\=OQ.
spo_qualifier([S,P,O,QP,QO], [22|W1]) :-
	spo_2([S,P,O], W1),
	gram(_,"acl:relcl",QO,QP),
	P \= QP,
	O\=QO.
spo_qualifier([S,P,O,QP,QO], [23|W1]) :-
	spo_2([S,P,O],W1),
	gram(_,"acl:relcl",QO,_),
	gram(_,"nmod",O,QO),
	gram(_,"case",QO,QP),
	O\=QO.
spo_qualifier([S,P,O,QP,OQ], [23|W1]) :-
	spo_2([S,P,O],W1),
	gram(_,"acl:relcl",OQ,QP),
	gram(_,"nmod",O,OQ),
	O\=OQ.
spo_qualifier([S,P2,O1,P1,O2], [24|W]) :-
	spo_2([S,P1,O1],W1),
	spo_2([S,P2,O2],W2),
	append(W1,W2,W),
	O1 \= O2,
	P1 \= P2.
spo_qualifier([S,P1,O,P2,O2],[24|W]) :-
	spo_2([S,P1,O],W1),
	spo_2([O,P2,O2],W2),
	append(W1,W2,W),
	P1\=P2,
	S\=O2,
	O\=O2.

spo_qualifier([S1,P1,O1,P2,S2],[25|W]) :-
	spo_2([S1,P1,O1],W1),
	spo_2([S2,P2,O2],W2),
	append(W1,W2,W),
	S1 \= S2,
	O1 \= O2,
	O1 \= S2,
	P1 \= P2,
	gram(_,"nmod",O1,O2),
	(_,"PRP")=S2.

spo_qualifier([S1,P1,O1,PQ,O2], [26|W]) :-
	spo_2([S1,P1,O1],W1),
	spo_2([S2,P2,O2],W2),
	append(W1,W2,W),
	S1 \= S2,
	O1 \= O2,
	P1 \= P2,
	gram(_,"case",O2,PQ),
	(PQT,_)=PQ,split_string(PQT,"-","",[PQTTerm|_]),
	member(PQTTerm,["for","to","through"]).
spo_qualifier([S1,P1,O1,PQ,S2], [26|W]) :-
	spo_2([S1,P1,O1],W1),
	spo_2([S2,P2,O2],W2),
	append(W1,W2,W),
	S1 \= S2,
	O1 \= O2,
	O1 \= S2,
	P1 \= P2,
	gram(_,"case",O2,PQ),
	(_,"PRP")=S2,
	(PQT,_)=PQ,split_string(PQT,"-","",[PQTTerm|_]),
	member(PQTTerm,["for","to","through"]).


%% test :-
%% 	[[S,_],[P,_],[O,_],[PQ,_],[OQ,_]]=[["cats-4",_],["went-11",_],["table-25",_],["drove-21",_],["store-15",_]],
%% 	spo_qualifier([[S,_],[P,_],[O,_],[PQ,_],[OQ,_]])
% phrase-pred-phrase-pred-phrase
% handles tokens and phrases since phrase_() return for both, single tokens and phase
%% spo_qualifier_phrase([SP,P,OP,PQ,OQP],	) :-
%% 	spo_qualifier([[S,_],[P,_],[O,_],[PQ,_],[OQ,_]],W),
%% 	S \= O,
%% 	O \= S,
%% 	O \= OQ,
%% 	SP=[S],
%% 	OP=[O],
%% 	OQP=[OQ].
%% 	%% phrase_([_,SP]),member(S,SP),
%% 	%% phrase_([_,OP]),member(O,OP),
%% 	%% phrase_([_,OQP]),member(OQ,OQP).
%% 	%% (phrase_([_,SP]),(last(SP,S) ; first(SP,S)) ; SP = [S]),
%% 	%% (phrase_([_,OP]),(last(OP,O) ; first(OP,O)) ; OP = [O]),
%% 	%% (phrase_([_,OQP]),(last(OQP,OQ) ; first(OQP,OQ)) ; OQP = [OQ]).


%% spo_phrase([SP,P,OP]) :-
%% 	spo_2([[S,_],P,[O,_]]),
%% 	phrase_([_,SP]),member(S,SP),
%% 	phrase_([_,OP]),member(O,OP).



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
phrase_([XI,[X]], [301]) :-
	(gram(XI,_,_,[X,XPOS]) ; gram(XI,_,[X,XPOS],_)),
	noun_type_tag(XPOS).

phrase_([XI,Phrase], [302]) :-
	(gram(XI,_,_,[X,XPOS]) ; gram(XI,_,[X,XPOS],_)),
	findall(Y,gram(_,"compound",[Y,_],[X,XPOS]),LSS),
	LSS\=[],
	Phrase=[X|LSS].

phrase_([XI,Phrase],[303]) :-
	(gram(XI,_,_,[X,XPOS]) ; gram(XI,_,[X,XPOS],_)),
	findall(Y,gram(_,"compound",[Y,_],[X,XPOS]),LSS),
	LSS\=[],
	conj([[X,XPOS],[CC,_]]),
	Phrase=[CC|LSS].

phrase_([XI,Phrase], [304]) :-
	(gram(XI,_,_,[X,XPOS]) ; gram(XI,_,[X,XPOS],_)),
	%% XPOS\="NNS",
	findall(Y,gram(_,"compound",[X,XPOS],[Y,_]),LSS),
	LSS\=[],
	conj([[KEY,_],[CC,_]]),
	member(KEY,LSS),
	[_|LSS2] = LSS,
	last(LSS2,KEY2),
	(gram(_,_,_,[KEY2,K2POS]) ; gram(_,_,[KEY2,K2POS],_)),
	K2POS\="NNS",
	append([CC|LSS2],[X],Phrase).

phrase_([XI,Phrase], [304]) :-
	(gram(_,_,_,[X,XPOS]) ; gram(_,_,[X,XPOS],_)),
	findall(Y,gram(_,"compound",[X,XPOS],[Y,_]),LSS),
	LSS\=[],
	[_|LSS2] = LSS,
	last(LSS2,KEY2),
	first(LSS,KEY3),
	(gram(_,_,_,[KEY2,K2POS]) ; gram(_,_,[KEY2,K2POS],_)),
	(gram(XI,_,_,[KEY3,_]) ; gram(XI,_,[KEY3],_)),
	K2POS\="NNS",
	append(LSS,[X],Phrase).
phrase_([XI,[CC|LSS]], [305]) :-
	(gram(XI,_,_,[X,XPOS]) ; gram(XI,_,[X,XPOS],_)),
	findall(Y,gram(_,"amod",[Y,_],[X,XPOS]),LSS),
	LSS\=[],
	conj([[X,XPOS],[CC,_]]).

% all nouns in a row
%% phrase_([XI,Phrase]) :-
%% 	(
%% 		(gram(XI,"compund",[X,XPOS],_) ; gram(_,"compund",_,[X,XPOS])),
%% 		noun_type_tag(XPOS),
%% 		term_num(X,XNUM)
%% 	),
%% 	findall([Y,YNUM],
%% 		(
%% 			(gram(_,_,[Y,YPOS],_) ; gram(_,_,_,[Y,YPOS])),
%% 			noun_type_tag(YPOS),
%% 			term_num(Y,YNUM),YNUM>XNUM
%% 		),
%% 		LTmp),

%% 	append([[X,XNUM]],LTmp,LIncr),
%% 	findall(I,member([_,I],LIncr),IList1),
%% 	remove_duplicates(IList1,IList),
%% 	write(IList),
%% 	incr_list(IList),
%% 	findall(V,member([V,_],LTmp),VList),
%% 	append([Y],VList,PhraseTmp),
%% 	append(PhraseTmp,[X],Phrase).


% TODO: sort phrases by number: abc-2,def-1 => def-1,abc-2
%% phrase_([XI,Phrase]) :-
%% 	gram(_,"compound",X,Y),
%% 	findall(Z,gram(_,"compound",Y,Z),L),L\=[],
%% 	append([Y,X],L,Phrase).

phrase_([XI,Phrase], [306]) :-
	gram(XI,"compound",[X,XPOS],[Y,_]),
	findall([Z,ZI],(gram(ZI,"compound",[X,XPOS],[Z,_]),ZI>XI),LTmp),
	append([[X,XI]],LTmp,LIncr),
	findall(I,member([_,I],LIncr),IList),incr_list(IList),
	findall(V,member([V,_],LTmp),VList),
	append([Y],VList,PhraseTmp),
	append(PhraseTmp,[X],Phrase).

phrase_([XI,Phrase], [307]) :-
	gram(XI,"amod",[X,_],[Y,_]),
	Phrase = [Y,X].
phrase_([XI,Phrase], [307]) :-
	gram(XI,"acl",[X,_],[Y,_]),
	Phrase = [X,Y].

% amod + compound phrase
phrase_([XI,Phrase], [308]) :-
	gram(XI,"amod",[X,XPOS],[Y,_]),
	findall([Z,ZI],(gram(ZI,"compound",[X,XPOS],[Z,_])),LTmp),
	findall(V,member([V,_],LTmp),VList),
	append([Y,X],VList,Phrase).

% amod + compound phrase
phrase_([XI,Phrase],[308]) :-
	gram(XI,"amod",[X,XPOS],[Y,_]),
	findall([Z,ZI],(gram(ZI,"compound",[Z,_],[X,XPOS])),LTmp),
	findall(V,member([V,_],LTmp),VList),
	append([Y,X],VList,Phrase).

% amod phrase
phrase_([XI,Phrase], [309]) :-
	gram(XI,"amod",[X,XPOS],_),
	findall([Z,ZI],(gram(ZI,"amod",[X,XPOS],[Z,_])),LTmp),
	findall(V,member([V,_],LTmp),VList),
	append(VList,[X],Phrase).
% conj + amod phrase
phrase_([XI,Phrase], [305]) :-
	gram(XI,"amod",[X,XPOS],_),
	findall([Z,ZI],(gram(ZI,"amod",[X,XPOS],[Z,_])),LTmp),
	findall(V,member([V,_],LTmp),VList1),
	VList1=[KEY|VList2],
	conj([[KEY,_],[CC,_]]),
	VList3=[CC|VList2],
	append(VList3,[X],Phrase).
	
phrase_([XI,Phrase], [310]) :-
	gram(XI,"nmod",[X,_],[Y,_]),
	gram(_,"case",[Y,_],[Z,_]),
	Phrase = [X,Z,Y].
phrase_([XI,Phrase], [311]) :-
	gram(XI,"nmod",[X,XPOS],[Y,_]),
	findall([Z,ZI],(gram(ZI,"compound",[Z,_],[X,XPOS])),LTmp),
	findall(V,member([V,_],LTmp),VList),
	append([Y,X],VList,Phrase).
phrase_([XI,Phrase], [311]) :-
	gram(XI,"nmod",[X,XPOS],[Y,_]),
	findall([Z,ZI],(gram(ZI,"compound",[X,XPOS],[Z,_])),LTmp),
	findall(V,member([V,_],LTmp),VList),
	append([Y,X],VList,Phrase).

phrase_([YI,Phrase], [312]) :-
	conj([[X,XPOS],[Y,YPOS]]),
	(gram(YI,_,[Y,YPOS],_) ; gram(YI,_,_,[Y,YPOS])),
	gram(_,"acl",[X,XPOS],[Z,_]),
	Phrase=[Y,Z].


% dep is a fallback relation if one is not infered properly.
phrase_([XI,Phrase], [313]) :-
	gram(XI,"dep",[X,_],[Y,_]),
	Phrase=[Y,X].


mod_([A,B],[430]) :-
	gram(_,"amod", A,B).
mod_([A,B],[431]) :-
	gram(_,"acl", A,B).
mod_([A,B],[432]) :-
	gram(_,"acl:relcl", A,B).
phrase_([XI,Phrase],[314]) :-
	mod_([[A,_],_],_),
	(gram(XI,_,[A,_],_);gram(XI,_,_,[A,_])),
	findall(C, mod_([[A,_],[C,_]],_), L), append(L,[A],Phrase).



phrase_pos([XI,L2],W) :-
	phrase_([XI,Phrase],W), 
	findall([P,POS],(member(P,Phrase),(gram(_,_,[P,POS],_);gram(_,_,_,[P,POS]))), L),
	remove_duplicates(L,L2).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pnoun_tag("NNP").
noun_tag("NN") :- !.
noun_tag(TAG) :-
	string_length(TAG,3),
	sub_string(TAG,0,2,_,"NN"),
	not(pnoun_tag(TAG)).

noun([XI,X]) :-
	gram(XI,_,[X,XPOS],_),
	noun_tag(XPOS).
noun([XI,X]) :-
	gram(XI,_,_,[X,XPOS]),
	noun_tag(XPOS).
pnoun([XI,X]) :-
	gram(XI,_,[X,XPOS],_),
	pnoun_tag(XPOS).
pnoun([XI,X]) :-
	gram(XI,_,_,[X,XPOS]),
	pnoun_tag(XPOS).

noun_phrase([XI,Phrase]) :-
	gram(XI,"compound",[X,XPOS],[Y,YPOS]),
	noun_tag(XPOS),noun_tag(YPOS),
	findall([Z,ZI],
		(gram(ZI,"compound",[X,XPOS],[Z,ZPOS]),
		 noun_tag(ZPOS),
		 ZI>XI
		),LTmp),
	append([[X,XI]],LTmp,LIncr),
	findall(I,member([_,I],LIncr),IList),incr_list(IList),
	findall(V,member([V,_],LTmp),VList),
	append([Y],VList,PhraseTmp),
	append(PhraseTmp,[X],Phrase).

noun_phrase([XI,Phrase]) :-
	phrase_([_,Phrase]),
	last(Phrase,X),
	gram(XI,"amod",[X,XPOS],[Y,_]),
	noun_tag(XPOS),
	findall([Z,ZI],(gram(ZI,"compound",[Z,_],[X,XPOS])),LTmp),
	findall(V,member([V,_],LTmp),VList),
	append([Y,X],VList,Phrase).

noun_phrase([XI,Phrase]) :-
	gram(XI,"amod",[X,XPOS],[Y,_]),
	noun_tag(XPOS),
	findall([Z,ZI],(gram(ZI,"compound",[Z,_],[X,XPOS])),LTmp),
	findall(V,member([V,_],LTmp),VList),
	append([Y,X],VList,Phrase).


pnoun_phrase([XI,Phrase]) :-
	gram(XI,"compound",[X,XPOS],[Y,YPOS]),
	pnoun_tag(XPOS),pnoun_tag(YPOS),
	findall([Z,ZI],
		(gram(ZI,"compound",[X,XPOS],[Z,ZPOS]),
		 pnoun_tag(ZPOS),
		 ZI>XI
		),LTmp),
	append([[X,XI]],LTmp,LIncr),
	findall(I,member([_,I],LIncr),IList),incr_list(IList),
	findall(V,member([V,_],LTmp),VList),
	append([Y],VList,PhraseTmp),
	append(PhraseTmp,[X],Phrase).

noun_or_phrase([XI,X]) :-
	(length(X,1),[X1|_]=X,noun([XI,X1])) ;
	noun_phrase([XI,X]).
pnoun_or_phrase([XI,X]) :-
	(length(X,1),[X1|_]=X,pnoun([XI,X1])) ;
	pnoun_phrase([XI,X]).
prp_or_phrase([XI,X]) :-
	phrase_([XI,X],_),
	[X1|_]=X,
	(gram(_,_,(X1,XPOS),_) ; gram(_,_,_,[X1,XPOS])),
	prp_tag(XPOS).

verb_tag("VB") :- !.
verb_tag(TAG) :-
	string_length(TAG,3),
	sub_string(TAG,0,2,_,"VB"),
	not(pnoun_tag(TAG)).
verb([XI,X]) :-
	gram(XI,_,[X,XPOS],_),
	verb_tag(XPOS).
verb([XI,X]) :-
	gram(XI,_,_,[X,XPOS]),
	verb_tag(XPOS).
prp_tag("PRP") :- !.
prp_tag("PRP$") :- !.

noun_type_tag("NN").  %	Noun, singular or mass
noun_type_tag("NNS").  %	Noun, plural
noun_type_tag("NNP").  %	Proper noun, singular
noun_type_tag("NNPS").  %	Proper noun, plural
noun_type_tag("PDT").  %	Predeterminer
noun_type_tag("POS").  %	Possessive ending
noun_type_tag("PRP").  %	Personal pronoun
noun_type_tag("PRP$").  %	Possessive pronoun
noun_type_tag("WDT").  %	Wh-determiner
noun_type_tag("WP").   %	Wh-pronoun
noun_type_tag("WP$").  %"	Possessive wh-pronoun
noun_type_tag("WRB").  %	Wh-adverb


direction([S,P,O,D],[400|W]) :-
	spo_2([[S,_],[P,_],[O,_]],W),
	gram(_,_,[P,_],[D,DPOS]),
	member(DPOS,["TO","IN"]).

%TODO: handle "'ve"
%% focus([X,Y]) :-
%% 	coref([X,Y]).
focus([A],[410]) :-
	(gram(_,_,[A,_],_) ; gram(_,_,_,[A,_])),
	split_string(A,"-","",[ATerm|_]),
	member(ATerm,["he","she","they","them","their","him","her","this","these","those"]).

focus([X],[411]) :-
	(gram(_,_,[X,XPOS],_);gram(_,_,_,[X,XPOS])),
	prp_tag(XPOS).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% depth first search on conjuntions and co-references
spo_2([S,P,O],[31|W]) :-
	spo_1([S,P,O],W).
spo_2([S2,P2,O2],[32|W]) :-
	spo_1([S,P,O],W),
	route(S,S2,_),
	route(P,P2,_),
	route(O,O2,_),
	(S\=S2;P\=P2;O\=O2), 
	S2\=P2,S2\=O2,P2\=O2.

route(X,X,[]).
route(X,Y,R) :-
	route(X,Y,[X],R).
route(X,Y,_,[drive(X,Y)]) :-
	travel(X,Y).
route(X,Y,V,[drive(X,Z)|R]) :-
	travel(X,Z),
	\+ member(Z,V),
	route(Z,Y,[Z|V],R),
	Z \= Y.
%% travel is unidirectional
%% travel(FROM,TO) :- conj([TO, FROM]).
%% travel(FROM,TO) :- coref([TO,FROM]).
travel(FROM,TO) :- conj([FROM,TO]).
travel(FROM,TO) :- coref([TO,FROM]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% number rules: e.g. age, dollars
num_1(NUM) :-
	gram(_,_,_,NUM),
	NUM=[_,"CD"].
num([UNIT],NUM) :-
	num_1(NUM),
	gram(_,"nummod",UNIT,NUM),
	not(gram(_,"nmod",_,UNIT)).
num([MOD,UNIT],NUM) :-
	num_1(NUM),
	gram(_,"nummod",UNIT,NUM),
	gram(_,"nmod",MOD,UNIT).

%% testDIFF(DIFF) :- 
%% findall(X,spo_2old(X,W), L), remove_duplicates(L,L2),length(L,LL),length(L2,LL2), findall(_,(member(XX,L2),writeln([2,XX])),_),
%% findall(X,spo_3(X,W), L3), remove_duplicates(L3,L23),length(L3,LL3),length(L23,LL23),  findall(_,(member(XX,L23),writeln([3,XX])),_),
%% writeln("=============================="),
%% diff(L2,L23,DIFF),findall(_,(member(XX,DIFF),writeln([4,XX])),_),
%% subtract(L2,L23,R1),findall(_,(member(XX,R1),writeln([42,XX])),_),
%% subtract(L23,L2,R2),findall(_,(member(XX,R2),writeln([43,XX])),_)
%% .