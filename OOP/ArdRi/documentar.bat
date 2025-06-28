:: Este fichero genera la documentación, debe ser ejecutado dentro de la carpeta con el codigo (dentro de src)
:: Genera un folder llamado "doc" y dentro de el, un HTML con la documentación generada
javadoc -author -version -private -encoding UTF-8 -charset UTF-8 -sourcepath .\src\ -d doc -classpath .\lib\* -link https://docs.oracle.com/en/java/javase/20/docs/api/ -subpackages tafl

@echo off
pause