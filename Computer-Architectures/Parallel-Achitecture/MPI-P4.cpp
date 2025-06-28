/*
	Autor: Adrián Zamora Sánchez
	Fecha: 09/11/2023
	Uso: mpiexec.exe -n X MPI-Practica-9.exe N
		- Donde X puede ser cualquier numero de procesos > 2
		- Donde N puede ser cualquier tamaño de la matriz cuadrada
	Descripción: Programa que multiplica dos matrices cuadradas de tamaño NxN
	para ello se vale del paralelismo a nivel de proceso utilizando MPI, el proceso
	root (rango 0) envia los datos a los demás procesos y estos realizan los cálculos
	y devuelven los resultados al proceso root.
*/

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
	// Comprueba que se pasen los argumentos necesarios para ejecutar el programa
	if (argc < 2) {
		// Se explica la causa de la finalización del proceso con codigo de error
		printf("Se requiere al menos un argumento.\n");
		return 1;
	}

	// Variables de los procesos MPI
	int mi_rango, tamanoProcesos;

	// Solicita el tamaño de las matrices cuadradas
	int tamanoMatriz;
	tamanoMatriz = atoi(argv[1]);

	// Tamaño completo de la matriz
	int tamanoMatrizCompleta = tamanoMatriz * tamanoMatriz;

	// Inicializamos el entorno MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &mi_rango);
	MPI_Comm_size(MPI_COMM_WORLD, &tamanoProcesos);

	/*
		Si se lanzan más procesos que filas tiene la matriz, el programa entrará en un bloqueo siempre,
		esto ocurre porque, si los procesos se aislan del proceso de recogida y calculo, entonces el proceso
		root (rango = 0) no podrá terminar la operación de broadcast, resultando en un bloqueo.

		Si por otro lado se permite que estos procesos lleguen a recibir el broadcast (además de ser ineficiente),
		estos van a llegar al bucle donde esperan un mensaje del root, puesto que su rango > num filas, estos nunca
		recibiran una fila con la que operar, pues ya se ha repartido y operado con todas ellas.

		Para evitar comportamientos inesperados y/o perdidas de rendimiento, se abortan los procesos sobrantes
		y se deja un comunicador nuevo con los procesos necesarios para realiar la tarea.
	*/

	MPI_Group grupoOriginal, nuevoGrupo;
	MPI_Comm comunicadorNuevo = MPI_COMM_WORLD;

	if (tamanoProcesos > tamanoMatriz) {
		// Numero de procesos que se pueden utilizar en caso de que sobren procesos
		int tamanoProcesosUtiles = tamanoMatriz + 1; // El proceso "+1" es el P0 que no participa en los calculos pero es necesario

		// Array que contendrá rangos de procsos
		int* rangos;
		rangos = (int*)malloc((tamanoProcesosUtiles) * sizeof(int));

		// Separa en el array rangos los procesos que se utilizarán
		for (int i = 0; i < tamanoProcesosUtiles; i++) {
			rangos[i] = i;
		}
		
		// Se recogen todos los procesos de comm world
		MPI_Comm_group(MPI_COMM_WORLD, &grupoOriginal); 
		
		// Crear un nuevo grupo con los procesos especificados
		MPI_Group_incl(grupoOriginal, tamanoProcesosUtiles, rangos, &nuevoGrupo);
		
		// Crear el nuevo comunicador
		MPI_Comm_create(MPI_COMM_WORLD, nuevoGrupo, &comunicadorNuevo); 
	}

	// Se finalizan los procesos que no se utilizan nunca
	if (mi_rango > tamanoMatriz) {
		MPI_Finalize();
		return 0;
	}

	// Se recogen de nuevo los rangos y tamaño del nuevo comunicador
	MPI_Comm_rank(comunicadorNuevo, &mi_rango);
	MPI_Comm_size(comunicadorNuevo, &tamanoProcesos);

	// Se comprueba que haya al menos 2 procesos
	if (tamanoProcesos < 2) {
		printf("Este programa trata sobre paralelizacion, por lo tanto necesita al menos 2 procesos para funcionar.\n");
		MPI_Finalize();
		return 1;
	}

	// Declara las matrices como arreglos de punteros
	float** matrizB;
	matrizB = (float**)malloc(tamanoMatriz * sizeof(float*));

	// Se declaran punteros tipo float
	float* Mf2;

	// Los punteros apuntan a bloques de memoria contigua capaces de almacenar el tamaño completo de las matrices
	Mf2 = (float*)malloc(tamanoMatrizCompleta * sizeof(float));

	// Se asigna toda la memoria de cada matriz de forma contigua
	for (int i = 0; i < tamanoMatriz; i++) {
		matrizB[i] = Mf2 + i * tamanoMatriz;
	}

	// Variables de control de las comunicaciones
	MPI_Status status;
	MPI_Request request;

	// El proceso 0 rellena las matrices
	if (mi_rango == 0) {
		/*
			Se crean las matrices A y C en el proceso 0 al igual que se ha creado la B anteriormente
			esto se hace en en proceso 0 únicamente con el objetivo de no reservar memoria de estas dos matrices
			en los otros procesos, pues no los van a usar y disparía el uso de memoria del programa
		*/

		// Declara las matrices como arreglos de punteros
		float** matrizA;
		float** matrizC;
		matrizA = (float**)malloc(tamanoMatriz * sizeof(float*));
		matrizC = (float**)malloc(tamanoMatriz * sizeof(float*));

		// Se declaran punteros tipo float
		float* Mf1;
		float* Mf3;

		// Los punteros apuntan a bloques de memoria contigua capaces de almacenar el tamaño completo de las matrices
		Mf1 = (float*)malloc(tamanoMatrizCompleta * sizeof(float));
		Mf3 = (float*)malloc(tamanoMatrizCompleta * sizeof(float));

		// Se asigna toda la memoria de cada matriz de forma contigua
		for (int i = 0; i < tamanoMatriz; i++) {
			matrizA[i] = Mf1 + i * tamanoMatriz;
			matrizC[i] = Mf3 + i * tamanoMatriz;
		}

		// Generamos una semilla de numeros aleatorios
		srand(time(NULL));

		// Los elementos de las matrices A y B se inicializan con float aleatorios
		for (int i = 0; i < tamanoMatriz; i++) {
			for (int j = 0; j < tamanoMatriz; j++) {
				matrizA[i][j] = ((float)rand()) / RAND_MAX;
				matrizB[i][j] = ((float)rand()) / RAND_MAX;
			}
		}

		// Muestra los valores de las matrices
		printf("Matriz A:\n");
		for (int i = 0; i < tamanoMatriz; i++) {
			for (int j = 0; j < tamanoMatriz; j++) {
				printf("%.3f  ", matrizA[i][j]);
			}
			printf("\n");
		}

		// Muestra los valores de las matrices
		printf("Matriz B:\n");
		for (int i = 0; i < tamanoMatriz; i++) {
			for (int j = 0; j < tamanoMatriz; j++) {
				printf("%.3f  ", matrizB[i][j]);
			}
			printf("\n");
		}

		// Se toma una marca de tiempo del momento donde empieza el calculo de la matriz (incluye envíos)
		double tiempoInicio = MPI_Wtime();

		// El proceso 0 pasa a todos los procesos la matriz B completa
		MPI_Bcast(matrizB[0], tamanoMatrizCompleta, MPI_FLOAT, mi_rango, comunicadorNuevo);

		// Envío de cada fila de la matriz
		int filasEnviadas = 0;
		int filasRecibidas = 0;

		// Variable que almacena la fila recibida
		float* fila = (float*)malloc(tamanoMatriz * sizeof(float));

		while (filasEnviadas < tamanoMatriz) {
			for (int i = 1; i < tamanoProcesos; i++) {
				// Control para no enviar filas superiores al tamaño de la matriz
				if (filasEnviadas >= tamanoMatriz) {
					break;
				}

				// Envío de datos
				MPI_Isend(matrizA[filasEnviadas], tamanoMatriz, MPI_FLOAT, i, filasEnviadas, comunicadorNuevo, &request);

				// Se incrementa el contador para enviar la proxima fila
				filasEnviadas++;

			}

			for (int i = 1; i < tamanoProcesos; i++) {
				// Control para que no se espere un retorno de valores si ya se han recibido todas la filas
				if (filasRecibidas >= tamanoMatriz) {
					break;
				}

				// Recibida de datos
				MPI_Recv(fila, tamanoMatriz, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, comunicadorNuevo, &status);

				// Se ordenan los datos recibidos en la matriz de resultados
				for (int j = 0; j < tamanoMatriz; j++) {
					matrizC[status.MPI_TAG][j] = fila[j];
				}

				// Se incrementa el contador de filas recibidas
				filasRecibidas++;
			}
		}

		// Se toma una marca de tiempo del momento donde finaliza el proceso paralelo
		double tiempoFin = MPI_Wtime();

		// Se muestran los resultados del calculo 
		printf("Matriz resultado: \n");
		for (int i = 0; i < tamanoMatriz; i++) {
			for (int j = 0; j < tamanoMatriz; j++) {
				printf("%.3f ", matrizC[i][j]);
			}
			printf("\n");
		}

		// Muestra el tiempo empleado (en segundos)
		printf("Se ha tardado %.9f\n", tiempoFin - tiempoInicio);

		// Se libera la memoria de la variable fila
		free(fila);

		// Se liberan todas las reservas de las matrices A y C del proceso 0
		free(Mf1);
		free(Mf3);
		free(matrizA);
		free(matrizC);
	}
	else {
		// Se llama al broadcast para que todos los procesos lo reciban
		MPI_Bcast(matrizB[0], tamanoMatrizCompleta, MPI_FLOAT, 0, comunicadorNuevo);

		// Variable que almacena la fila recibida
		float* fila = (float*)malloc(tamanoMatriz * sizeof(float));

		// Variable que almacena los resultados
		float* resultado = (float*)malloc(tamanoMatriz * sizeof(float));

		// Bucle infinito que recibe las matrices de forma asincrona las filas y hace los cálculos
		while (1) {
			// Variable de control para salir del bucle infinito
			bool finalizar = false;

			// recibe la fila de la matriz
			MPI_Recv(fila, tamanoMatriz, MPI_FLOAT, 0, MPI_ANY_TAG, comunicadorNuevo, &status);

			// Se comprueba que deba finalizar 
			// Este caso se da cuando este proceso + los otros trabajadores completan todas las filas restantes
			if ((status.MPI_TAG + tamanoProcesos - 1) >= tamanoMatriz) {
				finalizar = true;
			}

			// Inicializar el array resultado
			for (int i = 0; i < tamanoMatriz; i++) {
				resultado[i] = 0;
			}

			// Calcula el resultado de cada fila * columna
			for (int i = 0; i < tamanoMatriz; i++) {
				for (int j = 0; j < tamanoMatriz; j++) {
					resultado[j] += fila[i] * matrizB[i][j];
				}
			}

			// Envía de vuelta al proceso 0 la fila calculada
			MPI_Send(resultado, tamanoMatriz, MPI_FLOAT, 0, status.MPI_TAG, comunicadorNuevo);

			// Finaliza el bucle infinito
			if (finalizar == true) {
				break;
			}
		}

		// Se liberan las reservas de las variables usadas en el bucle anterior
		free(fila);
		free(resultado);
	}

	// Se libera la reserva de la matriz B de todos los procesos
	free(Mf2);
	free(matrizB);

	// Finalizamos el entorno MPI y la rutina main
	MPI_Finalize();

	return 0;
}
