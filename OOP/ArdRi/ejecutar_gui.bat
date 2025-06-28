:: Autor: Adrián Zamora Sánchez
:: Script que ejectura el programa en modo gráfico

:: Se pide el modo de juego al usuario por consola
@echo off
set /p userInput=Por favor, introduce un el tipo de juego Brandubh o ArdRi: 

:: Se espera la exitencia de la librería brandubh-gui-lib-1.0.0.jar y 
:: una versión java compatible con Java Runtime (class file version 64.0) o más reciente
java -Dfile.encoding=UTF-8 -classpath ".\bin;.\lib\tafl-gui-lib-1.0.0.jar;" tafl.gui.Tafl %userInput%

@echo off
pause