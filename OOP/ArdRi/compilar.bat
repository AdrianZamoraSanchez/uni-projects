:: Autor: Adrián Zamora Sánchez
:: Script que compila el programa a partir de los ficheros dentro del directorio src

:: Compila en orden para coincidir con las dependencias
javac -d bin src\tafl\util\*.java
javac -d bin -cp bin src\tafl\modelo\*.java
javac -d bin -cp bin src\tafl\control\*.java
javac -d bin -cp bin src\tafl\textui\*.java

@echo off
pause