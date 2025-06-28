/*
* Autor: Adrián Zamora Sánchez
* Fecha: 26/09/2023
* Ejecución con: "mpiexec.exe -n 2 MPI-Practica-6.exe"
* 
* Descripción: Programa que cálcula factoriales,
* para ello un proceso 0 pide los numeros de los cuales
* calcular el factorial y recibe y muestra los resultados
* en paralelo un proceso 1 calcula los factoriales.
*/

#include <mpi.h>
#include <iostream>

int main(int argc, char* argv[])
{
    // Declaración de las variables del programa
    int mirango, dato = 0, resultado = 0;

    // Inicialización de MPI
    MPI_Init(&argc, &argv);
    // Almacena el rango de cada proceso en la variable mirango
    MPI_Comm_rank(MPI_COMM_WORLD, &mirango);   

    // Crea dos variabes que controlan la comunicación entre procesos
    MPI_Request request_send, request_recv;

    // Bucle que mantiene funcionando ambos procesos hasta encontrar un 0
    while (true){
        // Toma de datos y salida de resultados por el proceso 0
        if (mirango == 0) {
                std::cout << "Proceso " << mirango << ": Introduce un numero para calcular su factorial: ";
                std::cin >> dato;

                // Envía el dato al proceso 1 para su cálculo
                MPI_Isend(&dato, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &request_send);
                std::cout << "Proceso " << mirango << ": numero(" << dato << ") enviado!\n";

                // Comprueba la condición de salida
                if (dato == 0) {
                    break;
                }

                std::cout << "Proceso " << mirango << ": recibiendo resultados...\n";

                // Recibe el resultado del proceso 1
                MPI_Irecv(&resultado, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, &request_recv);
                // Espera a recibir el resultado
                MPI_Wait(&request_recv, MPI_STATUS_IGNORE); 

                // Muestra el resultado
                std::cout << "Proceso " << mirango << ": el resultado de: " << dato << "! = " << resultado << "\n";
        }
        // Cálculos realizados por el proceso 1
        if (mirango == 1) {
                // Se espera a recibir el numero del que queremos calcular su factorial
                MPI_Irecv(&dato, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request_recv);
                // Espera a recibir el numero
                MPI_Wait(&request_recv, MPI_STATUS_IGNORE); 

                // Comprueba la condición de salida
                if (dato == 0) {
                    break;
                }

                // Calcula el factorial
                int factorial = 1;
                for (int i = 1; i <= dato; ++i) {
                    factorial *= i;
                }

                // Envía el resultado al proceso 0
                MPI_Isend(&factorial, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &request_send);
        }
    }
    // Finaliza el programa
    MPI_Finalize();
    return 0;
}
