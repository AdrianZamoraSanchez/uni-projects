:: Autor: Adrián Zamora Sánchez
:: Script que compila el programa a partir de los ficheros dentro del directorio src

:: Compila en orden para coincidir con las dependencias
javac -d bin src\Brandubh\util\*.java
javac -d bin -cp bin src\Brandubh\modelo\*.java
javac -d bin -cp bin src\Brandubh\control\*.java
javac -d bin -cp bin src\Brandubh\textui\*.java