@echo off
REM Limpiar y construir el proyecto Maven sin ejecutar los tests
echo Limpiando y construyendo el proyecto Maven...
call mvn clean package -DskipTests
if %errorlevel% neq 0 (
    echo Error al construir el proyecto Maven
    pause
    exit /b %errorlevel%
)

REM Construir imagenes de Docker
echo Construyendo imágenes de Docker...
call docker-compose build
if %errorlevel% neq 0 (
    echo Error al construir las imágenes de Docker
    pause
    exit /b %errorlevel%
)

REM Levantar contenedores de Docker en segundo plano
echo Levantando contenedores con Docker Compose...
call docker-compose up -d
if %errorlevel% neq 0 (
    echo Error al levantar los contenedores
    pause
    exit /b %errorlevel%
)

REM Mostrar logs del contenedor de la aplicacion
echo Mostrando logs del contenedor de la aplicacion...
docker-compose logs -f app

echo Script ejecutado correctamente.
pause