/* Programa lexer.l 

   Se utiliza pasandole el archivo a analizar como un parámetro,
   por ejemplo: ./lexer ficheroEntrada.ini
*/
%{
#include <stdio.h>
#include <string.h>
#include <ctype.h>

// Datos temporales (se busca la mayor sección y sus datos, el resto no se guarda)
char seccionActual[256] = "";
char seccionMayor[256] = "";
int maxPropiedades = 0;
int propiedadesSecActual = 0;

// Datos que se registra para todas las secciones
int numSecciones = 0;
int numRutasUnix = 0;
int numRutasWin = 0;
int numFechas = 0;
int numBool = 0;
int numOtros = 0;

/** 
 * Función encargada del conteo de secciones y de seleccionar a la mayor de ellas
 *
 * @param *nombre nombre de la sección que se está analizando
 * @return void
 */
void nuevaSeccion(char *nombre){
	// Aumenta el contador de secciones
	numSecciones++;
	
	// Comprueba si la sección actual tiene más propiedades que la anterior
	if(propiedadesSecActual > maxPropiedades){
		// Si es así cambia el número maximo de propiedades encontradas en una sección
		maxPropiedades = propiedadesSecActual;

		// Copia el string de la sección actual a la sección mayor
		strcpy(seccionMayor,seccionActual);
	}
	
	// Copia la nueva sección en sección actual
	strcpy(seccionActual,nombre);

	// Se reinicia el contador de propiedades de la sección actual
	propiedadesSecActual = 0;
}

/**
 * Función totalPropiedades
 *
 * @return unsigned int
 */
unsigned int totalPropiedades(){
	return numRutasUnix + numRutasWin + numFechas + numBool + numOtros;
}
%}


%%
#.* {/* Se ignora cualquier linea que empieza con '#', es un comentario */}
;.* {/* Se ignora cualquier linea que empieza con ';', es un comentario */}

\n {/* Se ignoran saltos de lineas */}
[ \t]+ {/* Se ignoran tabulaciones y espacios */}

\[([a-zA-Z0-9. _#\-]+)\][ \t]*([#;].*)? { 
	// Se extrae la cadena, sustituyendo el último caracter ']' por un nulo '\0'
	yytext[yyleng - 1] = '\0';
	
	// Se analiza la nuevas sección encontrada
	nuevaSeccion(yytext+1);
}

[a-zA-Z0-9._#\-]+[ \t]*=[ \t]*\/[a-zA-Z0-9._/-]+[ \t]* {
	// Propiedad de ruta en formato Unix
	numRutasUnix++;
	propiedadesSecActual++;
}

[a-zA-Z0-9._#\-]+[ \t]*=[ \t]*[A-Z]:\\[a-zA-Z0-9._\\-]+[ \t]* {
	// Propiedad de ruta en formato Windows
	numRutasWin++;
	propiedadesSecActual++;
}

[a-zA-Z0-9._#\-]+[ \t]*=[ \t]*([0-9]{4}[-/][0-9]{2}[-/][0-9]{2})|([0-9]{2}[-/][0-9]{2}[-/][0-9]{4})[ \t]* {
	// Propiedad de fecha con formatos DDMMYYYY o YYYYMMDD, donde el separador de D,M,Y puede ser '-' o '/'
	numFechas++;
	propiedadesSecActual++;
}

[a-zA-Z0-9._#\-]+[ \t]*=[ \t]*(True|False|Yes|No|1|0)[ \t]* { 
	// Propiedad con formato booleano
	numBool++;
	propiedadesSecActual++;
}

[a-zA-Z0-9._#\-]+[ \t]*=[ \t]*[^ \t\n#;]+[ \t]*([#;].*)? { 
	// Propiedad que no cumple ninguno de los formatos anteriores
	numOtros++;
	propiedadesSecActual++;
}

. {/* Se ignora cualquier otra cosa */}
%%

/**
 * Rutina principal del programa, ejecuta un análisis de la entrada y devuelve los datos sobre esta
 *
 * @return int 0 en caso de exito, otro número en caso de error
 */ 
int main(int argc, char *argv[]) {
	// Abrir archivo si se proporciona
	if (argc > 1) {
		FILE *file = fopen(argv[1], "r");
		if (!file) {
			printf("Error al abrir el archivo \"%s\"\n", argv[1]);
			return 1;
		}
		// Se esta blece el fichero como entrada del texto
		yyin = file;
	} else {
		// Sin archivo de entrada se utiliza la entrada por defecto (stdin)
		yyin = stdin;
	}

	// Ejecuta el analizador léxico
	yylex();

	// Comprobar si la última sección fue la que tenía más propiedades
	if (propiedadesSecActual > maxPropiedades) {
		maxPropiedades = propiedadesSecActual;
		strcpy(seccionMayor, seccionActual);
	}

	// Imprimir las estadísticas
	printf("Número de secciones: %d\n", numSecciones);
	printf("Número total de propiedades: %d\n", totalPropiedades());
	printf("La sección <<%s>> es la sección con más propiedades, con %d propiedades)\n", seccionMayor, maxPropiedades);
	printf("Número total de propiedades con valores de tipo ruta: %d\n", numRutasUnix+numRutasWin);
	printf("\tRutas Unix %d\n", numRutasUnix);
	printf("\tRutas Windows %d\n", numRutasWin);
	printf("Número total de propiedades con valores de tipo fecha: %d\n", numFechas);
	printf("Número total de propiedades con valores booleanos: %d\n", numBool);
	printf("Número total de propiedades con otros tipos de valores: %d\n", numOtros);

	// Se cierra el fichero de entrada si se abrió
	if(argc > 1){
		fclose(yyin);
	}

	return 0;
}
