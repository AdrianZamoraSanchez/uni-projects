/*
    Por: Adrián Zamora Sánchez
    
    Programa que lee los tokens devueltos por el lexer
    y realiza el análisis sintáctico de los mismos

    Ha conseguido traducir a código intermedio todos los ejemplos 
    siendo compilado con la siguiente cadena de comandos:

    flex lexer.l
    bison -d parser.y
    gcc lex.yy.c parser.tab.c -o parser -lfl
    ./parser nombreFichero.cobol

    
    Se han utilizado las versiones:
    - FLEX: 2.6.4
    - BISON: 3.8.2 
*/

%{
// Imports
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parser.h"

// Definición de funciones y sus tipos
int yylex();
void yyerror(const char *s);

// Definición de la variable que controla las etiquetas de los bloques de código
int etiqueta = 0;

// 
extern FILE *yyin;
%}

%union{
    // El programa trabaja con números y strings
    int num;
    char* string;
}

%type <num> literal while else varying
%type <string> comparation asign3 asign1 atomic varyingAssign varyingInc

%left '+' '-'
%left '*' '/'

%token TK_PROGRAM
%token TK_BEGIN
%token TK_END

%token <string>ID
%token <string>CAD
%token <num>NUM

%token TK_MOVE
%token TK_ADD
%token TK_TO
%token TK_SUBTRACT
%token TK_FROM
%token TK_MULTIPLY
%token TK_DIVIDE
%token TK_BY
%token TK_GIVING

%token TK_ACCEPT
%token TK_DISPLAY

%token TK_WHILE
%token TK_VARYING
%token TK_DO

%token TK_IF
%token TK_THEN
%token TK_ELSE

%token TK_IS
%token TK_NOT
%token TK_GREATER
%token TK_EQUAL
%token TK_LESS
%token TK_THAN

%start program
%%
program:TK_PROGRAM ID '.' TK_BEGIN stmts TK_END;

stmts: stmt | stmt stmts ;

stmt: loop '.'
    | asign '.'
    | cond '.'
    | io '.'
    ;

loop: loopType;

loopType: while
          booleanExpr { printf("\tsifalsovea LBL%d\n", $1 + 1); }
          TK_DO stmts
          TK_END {
            // Muestra el salto a la siguiente etiqueta
            printf("\tvea LBL%d\n", $1);
            printf("LBL%d:\n", $1+1);
        } 
        |varying ID varyingAssign {
            // Referencia a la variable contador
            printf("\tvalori %s\n", $2);
            
            // Si el valor desde el que se incrementa FROM atomic es num se muestra
            // muete NUM, por el lado contrario si es un ID muestra valord ID
            char *testOri;
            strtol($3, &testOri, 10);
            
            // Se comprueba si se inicializa desde número o variable
            if(*testOri == '\0'){printf("\tmete %s\n", $3);}
            else{printf("\tvalord %s\n",$3);}

            // Se asigna la variable de control y muestra la etiqueta del bloque de código
            printf("\tasigna\n");
            printf("LBL%d:\n", etiqueta);
        }
        TK_TO atomic varyingInc TK_DO stmts {
            // Asgina el incremento
            printf("\tvalori %s\n", $2);
            printf("\tvalord %s\n", $2);

            // Al igual que antes se trata de convertir el incremento en un número
            // y si es posible entonces se debe inicializar el incremento desde ID u no num
            char *testInc;
            strtol($3, &testInc, 10);

            // Se comprueba si se inicializa desde num o ID
            if(*testInc == '\0'){printf("\tmete %s\n", $7);}
            else{printf("\tvalord %s\n",$7);}
 
            // Bloque de código con incremento del contador
            printf("\tadd\n");
            printf("\tasigna\n");
            printf("\tvalord %s\n", $2);
            printf("\tmete %s\n", $6);

            // Bloque que comprueba si se debe salir del bucle o se mantiene dentro
            printf("\tesmenor\n");
            printf("\tsiciertovea LBL%d\n", $1);
            etiqueta++;
        } TK_END;

varying: TK_VARYING { $$=etiqueta; }

varyingAssign: TK_FROM atomic{ $$=$2; } 
                |  { $$="1"; /*Si no inicia el contador se deja en 1 por defecto*/}
                ;

varyingInc: TK_BY atomic { $$=$2; } 
        | { $$="1"; /*Si no hay incremento definido devuelve incremento 1*/ }
        ;

while: TK_WHILE{
        // Etiqueta del código del bucle
        printf("LBL%d:\n", etiqueta); 
        
        /* Se asigna como retorno el valor de la etiqueta para
           poder usarle dentro del resto de partes del bucle */
        $$=etiqueta;
        etiqueta++; // Se incrementa en uno la etiqueta
     }

asign: asign1 ID { 
            // Se compureba si es un ADD o un MOVE
            if($1 == "add"){
                // Si es ADD se imprime el valor y ADD
                printf("\tvalord %s\n\tadd\n", $2);
            }
            
            // Se muestra la asignación
            printf("\tvalori %s\n\tswap\n\tasigna\n", $2);
        }
     | asign2 ID { printf("\tvalord %s\n\tswap\n\tsub\n\tvalori %s\n\tswap\n\tasigna\n", $2, $2); }
     | asign3 ID { printf("\t%s\n\tvalori %s\n\tswap\n\tasigna\n", $1, $2); }
     ;

asign1: TK_MOVE expr TK_TO { $$ = "move"; }
      | TK_ADD expr TK_TO { $$ = "add"; }
      ;

asign2: TK_SUBTRACT expr TK_FROM;

asign3: TK_MULTIPLY expr TK_BY expr TK_GIVING { $$ = "mul";}
      | TK_DIVIDE expr TK_BY expr TK_GIVING { $$ = "div"; }
      ;

cond: TK_IF 
    booleanExpr
    TK_THEN
    {
        // Aumenta la etiqueta
        etiqueta++;

        // Código de decisión sobre entrar al bloque de if
        printf("\tsifalsovea LBL%d\n", etiqueta);          
    }
    stmts
    else
    TK_END { 
        /* Si no hubo else, se debe aumentar una vez más la etiqueta
           y mostrar la etiqueta del siguiente código */
        if($6 == 0){printf("LBL%d:\n", etiqueta++);}
    }
    ;

else: TK_ELSE 
    { 
        // Bloque del código else
        printf("\tvea LBL%d\n", etiqueta+1);
        printf("LBL%d:\n", etiqueta);
    } 
    stmts { 
        // Etiqueta para el código de después del bucle
        printf("LBL%d:\n", etiqueta+1); 
        $$=1; // Para control posterior
        etiqueta++;
    }
    | {$$=0;}
    ;

expr: expr '+' expr { printf("\tadd\n"); }
    | expr '-' expr { printf("\tsub\n"); }
    | mult          { }
    ;

mult: mult '*' val  { printf("\tmul\n"); }
    | mult '/' val  { printf("\tdiv\n"); }
    | val            { }
    ;

val:  NUM { printf("\tmete %d\n", $1); }
    | ID {
        // Muestra el valor y libera el puntero
        printf("\tvalord %s\n", $1);
        free($1);
    }
    | '(' expr ')'  { }
    ;

io: TK_DISPLAY literal { printf("\tprint %d\n", $2); }
  | TK_ACCEPT ID   { 
        // Muestra el código intermedio y libera el puntero
        printf("\tlee %s\n", $2); 
        free($2); }
  ;

literal: ID { 
            /* Variable que controla cuantas veces se repite 
               este toquen en expresiones con , */
            $$ = 1; 

            // Muestra el valor y libera el puntero
            printf("\tvalord %s\n", $1);
            free($1);
        } 
        | ID ',' literal { 
            /* Se suma el valor anterior con +1 de este
               esto permite controlar cosas como ID,NUM,ID */
            $$ = 1 + $3;

            // Muestra el valor y libera el puntero
            printf("\tvalord %s\n", $1);
            free($1);
        }
        | NUM { 
            // Muestra el valor y devuelve 1
            $$ = 1;
            printf("\tmete %d\n", $1);
        }
        | NUM ',' literal {
            // Muestra el valor y devuelve la suma su valor con los anteriores
            $$ = 1 + $3;
            printf("\tmete %d\n", $1);
        }
        | CAD { 
            // Devuelve 1
            $$ = 1; 

            // Muestra la cadena y libera el puntero al string
            printf("\tmetecad %s\n", $1);
            free($1);
        }
        | CAD ',' literal {
            // Devuelve la suma de su valor con los anteriores
            $$ = 1 + $3;

            // Muestra el código de almacenar la cadena y libera el puntero 
            printf("\tmetecad %s\n", $1);
            free($1);
        }
        ;
    
atomic: ID  {
            // Se crea una nueva referencia al string $1 
            $$ = strdup($1); 
            
            // Se libera $1
            free($1);
      }
      | NUM {
            // Se asigna espacio para la nueva cadena
            $$ = malloc(20); 

            // Se genera una cadena con el contenido numérico de $1
            sprintf($$, "%d", $1);
      }
      ;

booleanExpr: expr TK_IS comparation expr { 
                // Muestra el operador lógicode esta expresión
                printf("%s",$3); 
           }
           | expr TK_IS TK_NOT comparation expr { 
                // Niega el siguiente operador lógico 
                printf("%s\tnot\n",$4); 
           }
           ;

comparation: TK_GREATER TK_THAN { $$ = "\tesmayor\n"; }
           | TK_LESS TK_THAN    { $$ = "\tesmenor\n"; }
           | TK_EQUAL TK_TO     { $$ = "\tesigual\n"; }
           ;
%%
   
int main(int argc, char *argv[]){
    /* Comprueba si los datos se pasan por un fichero como argumento o como
       texto por la stdin */
    if (argc == 1) {
            // Modo interactivo
            printf("Ingrese texto el texto a procesar (Ctrl+D para cerrar):\n");
            yyin = stdin; // Se pasa a flex la entrada estándar
            yyparse(); // Procesa la salida
        
    } else if (argc == 2) {
        // Modo leer desde archivo
        FILE *file = fopen(argv[1], "r");
        
        // Comprueba que se pueda leer el archivo, sino da error
        if (!file) {
            printf("Error al abrir el archivo");
            return 1;
        }

        yyin = file; // Asocia el archivo al lexer
        yyparse(); // Procesa la salida
        fclose(file); // Cierra el archivo
    } else {
        // No esta permitido el uso de más de un argumento, devuelve error
        fprintf(stderr, "Uso: %s [archivo]\n", argv[0]);
        return 1;
    }

    return 0;
}

void yyerror(const char *s){
    printf("Error: %s\n", s);
}
