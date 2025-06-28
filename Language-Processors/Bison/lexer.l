/*
    Programa lexer.l
    Hecho por: Adrián Zamora Sánchez

    Analizador léxico que genera los tokens los cuales procesa
    el analizador sintático parser.exe generado con BISON
*/

%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parser.h"
%}


%option noyywrap
num 0|[1-9][0-9]*
id [A-Z][0-9A-Z\-]*[A-Z0-9]*
cad \"([^\"']|'[^'])*\"|\'([^\"']|"[^"])*\'

%%
\*>.*           {} // Se ignora los comentarios que empiezan con *> hasta el final de linea
"      \*".*      {} // Se ignora cualquier linea que comienza con 6 espacios y un asterico

[+\-\*/] { return yytext[0]; }
[()]     { return yytext[0]; }



"PROGRAM"   { return TK_PROGRAM; }
"BEGIN"     { return TK_BEGIN; }
[,.]        { return yytext[0]; }
"END"       { return TK_END; }

"MOVE"      { return TK_MOVE; }
"ADD"       { return TK_ADD; }
"TO"        { return TK_TO; }
"SUBTRACT"  { return TK_SUBTRACT; }
"FROM"      { return TK_FROM; }
"MULTIPLY"  { return TK_MULTIPLY; }
"DIVIDE"    { return TK_DIVIDE; }
"BY"        { return TK_BY; }
"GIVING"    { return TK_GIVING; }

"ACCEPT"    { return TK_ACCEPT; }
"DISPLAY"   { return TK_DISPLAY; }

"WHILE"     { return TK_WHILE; }
"VARYING"   { return TK_VARYING; }
"DO"        { return TK_DO; }

"IF"        { return TK_IF; }
"THEN"      { return TK_THEN; }
"ELSE"      { return TK_ELSE; }

"IS"        { return TK_IS; }
"NOT"       { return TK_NOT; }
"GREATER"   { return TK_GREATER; }
"EQUAL"     { return TK_EQUAL; }
"LESS"      { return TK_LESS; }
"THAN"      { return TK_THAN; }

{num} { yylval.num = atoi(yytext); return NUM; }
{id}  { yylval.string = strdup(yytext); return ID; }
{cad} { yylval.string = strdup(yytext); return CAD; }

[ \t\n] ;

. ;
%%
