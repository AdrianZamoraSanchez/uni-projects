#include <iostream>
#include <mpi.h>

int main(int argc, char* argv[]) {
    // Variables con las que trabajan los procesos hijos
    int mi_rango, tamano;
    
    // Se inicia MPI
    MPI_Init(&argc, &argv);

    // Se declaran los comunicadores
    MPI_Comm parent, intracom;
    
    // Se genera un intercomunicador con el padre
    MPI_Comm_get_parent(&parent);

    // Se crea un intracomunicador desde el intercomunicador del padre
    MPI_Intercomm_merge(parent, 1, &intracom);

    // Tomamos los valores del rango y el tamaño del intracomunicador
    MPI_Comm_rank(intracom, &mi_rango);
    MPI_Comm_size(intracom, &tamano);

    // Código para los procesos hijos
    printf("HIJO > proceso %d de %d: soy un hijo!\n", mi_rango, tamano);

    // Todos los hijos reciben este mensaje del padre
    char mensajeDelPadre[30];
    MPI_Recv(&mensajeDelPadre, 30, MPI_CHAR, 0, 0, intracom, MPI_STATUS_IGNORE);
    printf("HIJO > proceso %d de %d: %s\n", mi_rango, tamano, mensajeDelPadre);

    // El proceso hijo de menor rango (con el rango 1), saluda a sus hermanos y al padre
    if (mi_rango == 1) {
        printf("HIJO > proceso %d de %d: Soy el hijo menor\n", mi_rango, tamano);
        char mensaje[35] = "Saludo del hermano menor";

        // El hermano menor manda mensajes a sus procesos hermanos
        for (int i = 2; i < tamano; i++) {
            MPI_Send(&mensaje, 35, MPI_CHAR, i, 0, intracom);
        }

        // El hermano menor manda un mensaje al proceso padre
        MPI_Send(&mensaje, 35, MPI_CHAR, 0, 0, intracom);
    }

    // Los procesos hermnos mayores reciben el mensaje del hermano menor
    if (mi_rango != 1 && mi_rango != 0) {
        char mensajeHermano[35];
        MPI_Recv(&mensajeHermano, 35, MPI_CHAR, 1, 0, intracom, MPI_STATUS_IGNORE);
        
        // Muestran el mensaje
        printf("HIJO > proceso %d de %d recibido: %s\n", mi_rango, tamano, mensajeHermano);
    }

    // Finalizamos el entorno MPI
    MPI_Finalize();

    return 0;
}
