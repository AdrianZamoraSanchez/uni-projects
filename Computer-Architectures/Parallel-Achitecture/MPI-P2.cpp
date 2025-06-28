/*
* Autor: Adrián Zamora Sánchez
* Fecha: 02/10/2023
* Ejecución con: "mpiexec.exe -n 3 MPI-Practica-7.exe" (-n x donde x debe ser mayor o igual a 3)
*
* Descripción: Programa que divide una matriz en su matriz,
* triangular superior e inferior y asigna cada una de estas
* a un proceso, para ello se utiliza un tipo de datos que lee
* solamente los datos necesarios aplicando para esto saltos entre
* los datos que queremos leer.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define N 5 // Tamaño de la matriz

int main(int argc, char* argv[]) {
    // Variables de los procesos MPI
    int mi_rango, tamano;

    // Inicializamos el entorno MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mi_rango);
    MPI_Comm_size(MPI_COMM_WORLD, &tamano);

    // Declaración de las matrices
    int **matriz;

    // Asignamos a la matriz su espacio
    matriz = (int**)malloc(N * sizeof(int*));

    // Declaramos una variable de reserva de memoria
    int *rm;

    // Reservamos un bloque de memoria contigua y suficientemente grande como para toda la matriz
    rm = (int*)malloc(N * N * sizeof(int));

    // Asignamos partes de ese bloque contiguo a cada fila de la matriz
    for (int i = 0; i < N; i++){
        matriz[i] = rm + i * N;
    }

    // Rellena la matriz con numeros aleatorios en el proceso 0 solamente
    if (mi_rango == 0) {
        srand(time(NULL));
        printf("Matriz %d:\n", mi_rango);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matriz[i][j] = rand() % 10;
                printf("%d  ", matriz[i][j]); // Muestra en la consola la matriz
            }
            printf("\n");
        }
    }
    else {
        // Inicializar la matriz en procesos 1 y 2 con ceros
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matriz[i][j] = 0;
            }
        }
    }

    /*
    Triangular inferior
    Para leer los datos:
        Leemos al principio solamente el primer numero, en las siguiente lineas leemos 1 más cada vez
        (la triangular inferior tiene la fila 1 con el primer elemento y en las siguiente comienza a haber
        un elemento más cada vez)

    Para saltar a la posición desde la que se leen los datos:
        Saltamos i*N bloques de tamaño 4 bytes, es decir en la fila 0 no saltamos ninguno,
        en las siguientes saltamos siempre hasta el primer elemento de la fila i (nº de bloques)* N (tamaño de bloque) * de tamaño int
    */

    // Definimos los arreglos con las caracteristicas de la matriz triangular superior
    int vectorLongTriangularInf[N]; // Longitud de las filas de la matriz
    MPI_Aint vectorDesplTriangularInf[N]; // Desplazamiento donde se empieza a leer
    MPI_Datatype vector_tipos[N]; // Vector con los tipos de datos (en ambas int)
    MPI_Datatype triangularInf; // Definimos como triangularInf este tipo

    // Esta matriz debe tener 1 elemento en la primera fila e ir aumentando hasta N en su N fila
    for (int i = 0; i < N; i++) {
        vectorLongTriangularInf[i] = i+1;
    }

    // El desplazamiento sigue un patron de saltar 0, N, i*N bloques de tamaño int
    for (int i = 0; i < N; i++) {
        vectorDesplTriangularInf[i] = i * N * sizeof(int);
    }

    // El vector de tipos es común tanto para ambas triangulares, solo tienen int
    for (int i = 0; i < N; i++) {
        vector_tipos[i] = MPI_INT;
    }

    // Se define este nuevo tipo
    MPI_Type_create_struct(N, vectorLongTriangularInf, vectorDesplTriangularInf, vector_tipos, &triangularInf);
    MPI_Type_commit(&triangularInf);

    /*
    Triangular superior
    Para leer los datos:
        Leemos al principio el tamaño de una fila entera y sucesivamente leemos un dato menos
        (la triangular superior tiene la fila 1 completa y en las inferiores comienza a haber 0s)
    
    Para saltar a la posición desde la que se leen los datos:
        Saltamos i*(N+1) bloques de tamaño 4 bytes, es decir en la fila 0 no saltamos ninguno, 
        en la fila 1 saltamos el primero y comenzamos a leer, en la fila 2 saltamos los 2 primeros... etc
    */
    
    // Definimos los arreglos con las caracteristicas de la matriz triangular superior
    int vectorLongTriangularSup[N]; // Longitud de las filas de la matriz
    MPI_Aint vectorDesplTriangularSup[N]; // Desplazamiento donde se empieza a leer
    MPI_Datatype triangularSup; // Definimos como triangularSup este tipo

    // Esta matriz debe tener N elementos en su primera fila y se reducen hasta 1 en su ultima fila
    for (int i = 0; i < N; i++) {
        vectorLongTriangularSup[i] = N - i;
    }

    // El desplazamiento sigue un patron de saltar 0, 1, i*(N+1) bloques de tamaño int
    for (int i = 0; i < N; i++) {
        vectorDesplTriangularSup[i] = i*(N+1)*sizeof(int);
    }
    
    MPI_Type_create_struct(N, vectorLongTriangularSup, vectorDesplTriangularSup, vector_tipos, &triangularSup);
    MPI_Type_commit(&triangularSup);
    

    // Proceso 0 envía la matriz a los procesos 1 y 2
    if (mi_rango == 0) {

        // Sen envía el contenido de matrizaux completo (la matriz trinagular inferior) al proceso 2
        MPI_Send(&(matriz[0][0]), 1, triangularInf, 2, 0, MPI_COMM_WORLD);
        printf("Proceso %d: Matriz triangular inferior enviada\n", mi_rango);

        // Sen envía el contenido de matrizaux completo (la matriz trinagular superior) al proceso 1
        MPI_Send(&(matriz[0][0]), 1, triangularSup, 1, 0, MPI_COMM_WORLD);
        printf("Proceso %d: Matriz triangular superior enviada\n", mi_rango);
    }

    // Proceso 1 recibe la matriz triangular superior
    if (mi_rango == 1) {

        printf("Proceso %d, matriz previa a recibir los datos:\n", mi_rango);

        // Imprime el contenido de la matriz
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d  ", matriz[i][j]);
            }
            printf("\n");
        }

        // Recibe los datos de la matriz triangular superior
        MPI_Recv(&(matriz[0][0]), 1, triangularSup, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Proceso %d, recibida la matriz triangular superior:\n", mi_rango);

        // Imprime el contenido de la matriz
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d  ", matriz[i][j]);
            }
            printf("\n");
        }

        printf("Proceso %d: Fin", mi_rango);
    }

    // Proceso 2 recibe la matriz triangular inferior
    if (mi_rango == 2) {
        printf("Proceso %d, matriz previa a recibir los datos:\n", mi_rango);

        // Imprime el contenido de la matriz
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d  ", matriz[i][j]);
            }
            printf("\n");
        }

        // Recibe los datos de la matriz triangular inferior
        MPI_Recv(&(matriz[0][0]), 1, triangularInf, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Proceso %d, recibida la matriz triangular inferior:\n", mi_rango);

        // Imprime el contenido de la matriz
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d  ", matriz[i][j]);
            }
            printf("\n");
        }

        printf("Proceso %d: Fin", mi_rango);
    }
    
    // Liberamos los punterios a las matrices y los bloques de memoria
    free(matriz);
    free(rm);

    // Liberamos el tipo matrizTriangular
    MPI_Type_free(&triangularInf);
    MPI_Type_free(&triangularSup);

    // Finalizamos el entorno MPI
    MPI_Finalize();

    return 0;
}
