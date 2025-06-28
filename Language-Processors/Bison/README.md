# Práctica de bison
## Cómo funciona este microcompilador
- Generar el analizador léxico a partir del código flex (archivo con extensión .l):
  
```flex analizador_lexico.l```
- Generar el analizador sintáctico a partir del código bison (archivo con extensión .y):
  
```bison -yd analizador_sintactico.y```
- Compilar y enlazar los códigos C generados anteriormente con gcc:
  
```gcc lex.yy.c y.tab.c -o nombre_ejecutable -lfl```

## Ejecución del programa
- Para ejecutar el programa y pasar el código a la entrada principal:
```./nombre_ejecutable```

- Ejecutar el análisis de un código pasado como parámetro:
```./nombre_ejecutable archivo_fuente.txt```
