%------------------------------
% lib.pl module
%
% Library of miscellaneous logic.
%------------------------------
:- module(lib, []).

same(X,X).


abs(I,J) :- (
  (I < 0) ->
    (J is -1*I)
    ;
    (J is I)
).

min(V1, V2, MIN_V) :- (
  V1 < V2 -> 
    MIN_V = V1
    ; 
    MIN_V = V2
).

max(V1, V2, MAX_V) :- (
  V1 > V2 -> 
    MAX_V = V1
    ; 
    MAX_V = V2
).




% this predicate doesn't return false on empty list, unlike built in last/2.
safeLast([], _) :- (!).
safeLast([Elem], Elem).
safeLast([_|Tail], Elem) :- last(Tail, Elem).

% utility method for checking type of value
checkType(E, OUT) :- (
  (var(E), OUT = 'var')   ; 
  (atom(E), OUT = 'atom') ;
  (integer(E), OUT = 'integer') ;
  (float(E), OUT = 'float') ; 
  (atomic(E), OUT = 'atomic') ;
  (compound(E), OUT = 'compound') ;
  (nonvar(E), OUT = 'nonvar') ;
  (number(E), OUT = 'number') ; 
  (OUT = 'unknown')
).

listLength([], N) :- (
  N = 0
).
listLength(LIST, NN) :- (
  [_|T] = LIST,
  listLength(T, N),
  NN is N + 1
).

between(I,J,K) :- (
  (I =< J) ->
    (I =< K, J >= K)
    ;
    (I >= K, J =< K)
).

minElement(E1, V1, E2, V2, MIN_E) :- (
  V1 < V2 -> 
    MIN_E = E1
    ; 
    MIN_E = E2
).

sqr(A, OUT) :- (
  OUT is A*A
).

sumList([], 0).
sumList([H|T], Sum) :-
   sum_list(T, Rest),
   Sum is H + Rest.

%--------------------------
% predicates used for input stream processing File and keyboard.
%--------------------------
readString(IN, OUT) :- (readString(IN, OUT, ' ')).
readString(IN, OUT, DELIMITER) :- (
  read_line_to_codes(IN,DATA), 
  stringTokens(DATA, OUT, DELIMITER)
).

stringTokens(IN, OUT) :- stringTokens(IN, OUT, ' ').
stringTokens(end_of_file, end_of_file, _).
stringTokens(IN, OUT, DELIMITER) :- (
  atom_codes(STRING, IN), 
  atomic_list_concat(OUT, DELIMITER, STRING)
).

prompt(Q, TOKENS) :- (prompt(Q, TOKENS, ' ')).
prompt(Q, TOKENS, DELIMITER) :- (
  write(Q),
  readString(user_input, TOKENS, DELIMITER)
).

% ensures IN is a number. Also converts from an atom to a number.
ensureNumber(IN, NUM) :- (
  number(IN) -> 
    NUM = IN ; atom_number(IN, NUM)  
).


% Concatinate list elements into string.
s([], '').
s(S, OUT) :- (
  [S1|REST] = S,
  s(REST, NEW),
  atom_concat(S1, NEW, OUT),
  !
).
s(S, S) :- !.


:- dynamic str/1.
str(S) :- (S = '').
setStr(S) :- (
  retractall(str(_)),
  assert(str(S))
).


% For debug purposes. Used for showing entire reasoning tree.
% debug(0) = off
% debug(1) = on
:- dynamic debug/1.
debug(D) :- (D is 0).
setDebug(D) :- (
  retractall(debug(_)),
  assert(debug(D))
).
debugOn :- setDebug(1).
debugOff :- setDebug(0).
out(S) :- (
  debug(1) -> (rpt(S)) ; (true)
).


%------------------------------------------
% Following code used for printing indented strings.
%------------------------------------------
:- dynamic idn/1.
idn(IDN) :- (IDN is 0).
setIDN(IDN) :- (
  retractall(idn(_)),
  assert(idn(IDN))
).
incI :- (incI(_)).
incI(IDNN) :- (
  idn(IDN),
  retractall(idn(_)),
  IDNN is IDN+2,
  assert(idn(IDNN))
).
dcrI :- (dcrI(_)).
dcrI(IDNN) :- (
  idn(IDN),
  retractall(idn(_)),
  IDNN is IDN-2,
  assert(idn(IDNN))
).

rpt(S) :- (
  idn(I),
  pad(I),
  s(S,O),
  write(O),nl
).

pad(0) :- !.
pad(I) :- (
  I < 0 -> 
   (setIDN(0), pad(0)) ;
  (
    write(' '),
    IN is I - 1,
    pad(IN)
  )
).
%------------------------------------------
