# Práctica de Flex

## Pasos para ejecutar uma práctica de Flex:
- Compilar el lexer con Lex:

```lex lexer.l```

- Compila el archivo C generado con el nombre lex.yy.c:

```gcc lex.yy.c -o lexer -lfl```

## Ejecuta el programa Una vez compilado, puedes ejecutarlo:
- Se analizar un texto ejecutando el programa y escribiendo en la entrada estándar:

```./lexer```

- Se puede analiar un fichero fuente pasandolo como argumento:
  
```./lexer < input.txt```
