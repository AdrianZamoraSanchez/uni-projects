:: Autor: Adrián Zamora Sánchez
:: Script que ejectura el programa en modo texto

:: Se pide el modo de juego al usuario por consola
@echo off
set /p userInput=Por favor, introduce un el tipo de juego Brandubh o ArdRi: 

:: Se espera una versión de java compatible con Java Runtime (class file version 64.0) o más reciente
java -Dfile.encoding=UTF-8 -classpath ".\bin;" tafl.textui.Tafl %userInput%

:: Una pausa en la ejecución para evitar que se cierre la consola y se pueda leer el ganador
@echo off
pause