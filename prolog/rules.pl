:- style_check(-discontiguous).


indexOf([Element|_], Element, 0). % We found the element
indexOf([_|Tail], Element, Index):-
  indexOf(Tail, Element, Index1), % Check in the tail of the list
  Index is Index1+1.  % and increment the resulting index

%% sublist(S,M,N,[_A|B]):- M>0, M<N, sublist(S,M-1,N-1,B).
%% sublist(S,M,N,[A|B]):- 0 is M, M<N, N2 is N-1, S=[A|D], sublist(D,0,N2,B).    
sublist(L, M, N, S) :-
    findall(E, (between(M, N, I), nth1(I, L, E)), S).
%% service --> verb,noun.
%% service --> noun,verb,noun.
%% service --> verb,noun_phrase.





noun_phrase2([X,DET,Y]) :-
	det((X,"NNP"),(DET,_)),compound((X,"NNP"),(Y,"NNP")),!.
noun_phrase2([X,Y]) :-
	compound((X,"NNP"),(Y,"NNP")),!.
noun_phrase2([X,Y,Z]) :-
	noun_phrase2([X|Y]),
	compound((X,"NNP"),(Z,"NNP")),!.

t1 --> s1,s2,s3.
t1(P) --> cap(P),s2,s3.
cap(P,_,[]) :- P=["SS"].
cap(P,_,[]) :- P=["DD"].
cap(P,_,[]) :- P=["OO"].
s1 --> ["s1"].
s2 --> ["s2"].
s3 --> ["s3"].

service --> adj,noun_phrase,!.
service --> adj,noun,!.
service --> noun_phrase,!.
service --> noun,!.
%% service --> noun,cconj,noun.
%% service --> noun,cconj,noun_phrase.
%% service --> noun_phrase,cconj,noun_phrase.
%% service --> noun.
%% service --> adj,noun_phrase.

%% service --> adj,noun,cconj,noun.
%% service --> noun,cconj,noun.
%% service --> adj,noun,cconj,noun_phrase.
%% service --> noun_phrase.

program --> det,propn.
program --> det,pnoun_phrase.
program --> propn.
program --> pnoun_phrase.
program --> pnoun_phrase,verb.
program --> pnoun_phrase,verb_phrase,!.

verb_phrase --> verb, noun_phrase,!.
verb_phrase --> verb, noun,!.
noun_phrase --> noun, noun.
noun_phrase --> noun, noun,noun.
%% pnoun_phrase --> propn, propn.
pnoun_phrase --> propn, propn.

description --> program,verb,service.


program_(P,[]) :-
	ngram(_,EP,P),program(P,[]),
	ngram(SV,EV,V),verb(V,[]),EP<SV,
	ngram(SS,_,S),service(S,[]),EV<SS.

ss([P,V,S]) :-
	ngram(_,EP,P),program(P,[]),
	ngram(SV,EV,V),verb(V,[]),EP<SV,
	ngram(SS,_,S),service(S,[]),EV<SS.
%% ss([P,V,S1]) :-
%% 	ngram(_,EP,P),program(P,[]),
%% 	ngram(SV,EV,V),verb(V,[]),EP<SV,
%% 	ngram(SS1,ES1,SN1),service(SN1,S1,_),EV<SS1,
%% 	ngram(SADJ,EADJ,ADJ),adj(ADJ,[]),ES1<SADJ,
%% 	ngram(SS2,_,S2),service(S2,[]),EADJ<SS2.
%% ss([P,V,S2]) :-
%% 	ngram(_,EP,P),program(P,[]),
%% 	ngram(SV,EV,V),verb(V,[]),EP<SV,
%% 	ngram(SS1,ES1,SN1),service(SN1,_,_),EV<SS1,
%% 	ngram(SADJ,EADJ,ADJ),adj(ADJ,[]),ES1<SADJ,
%% 	ngram(SS2,_,S2),service(S2,[]),EADJ<SS2.
logical_form([P,V,S]) :-
	ngram(_,EP,P),program(P,[]),
	ngram(SV,EV,V),verb(V,[]),EP<SV,
	ngram(SS,_,SN),service_conj(SN,S),EV<SS,length(S,LenS),LenS>0.


service_conj(T,S) :- conj_s(T,S,_).
service(T,S) :- conj_s(T,_,S).
conj_s(S,[E0],[E2]) :-
	nth0(0,S,E0), noun([E0],[]),
	nth0(1,S,E1), cconj([E1],[]),
	nth0(2,S,E2), noun([E2],[]).
conj_s(S,[E01,E02],[E2]) :-
	nth0(0,S,E01),nth0(1,S,E02), noun_phrase([E01,E02],[]),
	nth0(2,S,E1), cconj([E1],[]),
	nth0(3,S,E2), noun([E2],[]).
conj_s(S,[E0],[E21,E22]) :-
	nth0(0,S,E0), noun([E0],[]),
	nth0(1,S,E1), cconj([E1],[]),
	nth0(2,S,E21),nth0(3,S,E22), noun_phrase([E21,E22],[]).
conj_s(S,[E01,E02],[E21,E22]) :-
	nth0(0,S,E01),nth0(1,S,E02), noun_phrase([E01,E02],[]),
	nth0(1,S,E1), cconj([E1],[]),
	nth0(3,S,E21),nth0(4,S,E22), noun_phrase([E21,E22],[]).






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

