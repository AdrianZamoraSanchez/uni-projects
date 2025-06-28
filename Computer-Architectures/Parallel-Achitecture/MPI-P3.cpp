/*
* Autor: Adrián Zamora Sánchez
* Fecha: 24/10/2023
* Ejecución con: "mpiexec.exe -n 1 MPI-Practica-8.exe"
*
* Descripción: Un proceso padre lanzara una cantidad de procesos hijos
*              estos procesos se comunicaran mandandose mensajes entre ellos
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <mpi.h>

int main(int argc, char* argv[]) {
    // Variables de los procesos MPI
    int mi_rango, tamano;

    // Inicializamos el entorno MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mi_rango);
    MPI_Comm_size(MPI_COMM_WORLD, &tamano);

    // El proceso 0 lanza a los procesos hijos
    if (mi_rango == 0) {
        // Rutal al programa hijo
        char programa[] = "C:/Users/adria/source/repos/MPI-Practica-8-2/x64/Debug/MPI-Practica-8-2.exe"; 

        // Variable que almacena la cantidad de procesos hijos que se deben lanzar
        int hijos;

        // Comunicador que permitirá el paso de mensajes entre el COMM_WORLD de los hijos y el del padre
        MPI_Comm intercom, intracom;

        // Se toma el número de hijos que se va a lanzar
        printf("PADRE > proceso %d de %d, cuantos procesos hijos? ", mi_rango, tamano);
        std::cin >> hijos;

        // Se lanzan tantos procesos hijos como introduzca el usuario
        MPI_Comm_spawn(programa, MPI_ARGV_NULL, hijos, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &intercom, MPI_ERRCODES_IGNORE);

        // Se crea un intracomunicador desde el intercomunicador generado al crear los proceso hijos
        MPI_Intercomm_merge(intercom, 0, &intracom);

        // Se vuelven a tomar los valores de tamaño y rango con el nuevo comunicador
        MPI_Comm_size(intracom, &tamano);
        MPI_Comm_rank(intracom, &mi_rango);

        // Se muestra como la variable "tamano" toma el valor hijos+1 pues se han juntado los comunicadores
        printf("PADRE > proceso %d de %d, lanzando %d procesos hijos\n", mi_rango, tamano, hijos);
        
        // Se envía el mensaje del proceso padre a todos los hijos
        char mensajeParaHijos[30] = "Mensaje del proceso padre";
        for (int i = 1; i < hijos+1; i++) {
            MPI_Send(&mensajeParaHijos, 30, MPI_CHAR, i, 0, intracom);
        }

        // Se recibe e imprime el mensaje del proceso hijo de menor rango
        char mensajeHijoMenor[35];
        MPI_Recv(&mensajeHijoMenor, 35, MPI_CHAR, 1, 0, intracom, MPI_STATUS_IGNORE);

        printf("PADRE > proceso %d de %d, %s\n", mi_rango, tamano, mensajeHijoMenor);
    }

    // Finalizamos el entorno MPI
    MPI_Finalize();

    return 0;
}