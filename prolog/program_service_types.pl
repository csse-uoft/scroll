:- style_check(-discontiguous).
:- ensure_loaded(token_kinds).
:- use_module(lib).


offers_syn("provides").
offers_syn("offers").
offers_syn("offer").
offers_syn("provide").
offers_syn("provided").
offers_syn("offered").

% In case terms are indexed with a postfix (e.g. "offer-4"),
% remove the pistfix and check term.
is_offers_syn(IN) :-
	split_string(IN,"-","",[TERM|_]),
	offers_syn(TERM).

program(S, [421|W]) :-
	spo([[S,_],[P,_],[O,_]],W),
	pnoun([_,S]),
	noun([_,O]),
	is_offers_syn(P).
service(O, [422|W]) :-
	spo([[S,_],[P,_],[O,_]],W),
	pnoun([_,S]),
	noun([_,O]),
	is_offers_syn(P).

program_offers_service([S,'offers',O],[420|W]) :-
	spo([[S,_],[P,_],[O,_]],W),
	pnoun([_,S]),
	noun([_,O]),
	is_offers_syn(P).


state([[MOD,POS],T],430) :-
	gram(_,"amod", T,[MOD,POS]),
	(sub_string(POS,0,2,_,"VB") ; sub_string(POS,0,2,_,"JJ")).
xstate([MOD,T],430) :-
	gram(_,"nmod", T,MOD).



%% program_offers_service([S,'to',O]) :-
%% 	spo_phrase([S,P,O]),
%% 	pnoun_or_phrase([_,S]),
%% 	noun_or_phrase([_,O]),
%% 	gram(_,_,_,[PP,PPOS]),
%% 	is_offers_syn(P).


pp([S1,P1,O1,P2,O2]) :- 
	spo([S1,P1,O1],_), spo([O1,P2,O2],_),
	S1\=S2,
	S1\=O1,
	S1\=O2,
	S2\=O1,
	S2\=O2,
	O1\=O2,
	P1\=P2.
pp([S1,P1,O1,P2,O2]) :- 
	spo([S1,P1,O1],_), spo([S1,P2,O2],_),
	S1\=S2,
	S1\=O1,
	S1\=O2,
	S2\=O1,
	S2\=O2,
	O1\=O2.
pp2([S1,P1,O1,P2,O2]) :- spo([S1,P1,O1]), spo([S1,P2,O2]).
pp2([S1,P1,O1,P2,O2]) :- spo([S1,P1,O1]), spo([O1,P1,O2]),P1=P2.
pp2([S1,P1,O1,P2,O2]) :- spo([S1,P1,O1]), spo([S1,P1,O2]),P1=P2.
