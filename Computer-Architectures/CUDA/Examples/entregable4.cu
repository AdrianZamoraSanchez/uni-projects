/*
* ARQUITECTURA DE COMPUTADORES
* Hecho por: Adrián Zamora Sánchez y Adrián Alcalde Alzaga
* Ejercicio: Entregable 4 de CUDA
* Descripción: Aprovechar el paralelismo de una GPU para optimizar algoritmos.
* Ordenar de menor a mayor un vector de tamaño N (definido en tiempo de ejecución por 
* el  usuario)  con  elementos  aleatorios  comprendidos  entre  1  y  30.  La  ordenación  debe 
* realizarse utilizando el método de ordenación por  rango, donde cada hilo de ejecución del 
* kernel debe encargarse de calcular el  rango de un único elemento y colocarlo en el vector final. 
*/

// includes
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <device_launch_parameters.h>

//defines
#define NBLOQUES 32
#define NHILOS 32


void propiedades_Device(int deviceID)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);

	// calculo del numero de cores (SP)
	int cudaCores = 0;
	int SM = deviceProp.multiProcessorCount;
	int maxThreads = deviceProp.maxThreadsPerBlock;
	int major = deviceProp.major;
	int minor = deviceProp.minor;
	const char* archName;

	switch (major)
	{
	case 1:
		//TESLA
		archName = "TESLA";
		cudaCores = 8;
		break;
	case 2:
		//FERMI
		archName = "FERMI";
		if (minor == 0)
			cudaCores = 32;
		else
			cudaCores = 48;
		break;
	case 3:
		//KEPLER
		archName = "KEPLER";
		cudaCores = 192;
		break;
	case 5:
		//MAXWELL
		archName = "MAXWELL";
		cudaCores = 128;
		break;
	case 6:
		//PASCAL
		archName = "PASCAL";
		cudaCores = 64;
		break;
	case 7:
		//VOLTA(7.0) //TURING(7.5)
		cudaCores = 64;
		if (minor == 0)
			archName = "VOLTA";
		else
			archName = "TURING";
		break;
	case 8:
		// AMPERE
		archName = "AMPERE";
		cudaCores = 64;
		break;
	case 9:
		//HOPPER
		archName = "HOPPER";
		cudaCores = 64;
		break;
	default:
		//ARQUITECTURA DESCONOCIDA
		archName = "DESCONOCIDA";
	}

	int rtV;
	cudaRuntimeGetVersion(&rtV);

	// presentacion de propiedades
	printf("***************************************************\n");
	printf("DEVICE %d: %s\n", deviceID, deviceProp.name);
	printf("***************************************************\n");
	printf("> CUDA Toolkit\t\t\t\t: %d.%d\n", rtV / 1000, (rtV % 1000) / 10);
	printf("> Arquitectura CUDA\t\t\t: %s\n", archName);
	printf("> Capacidad de Computo\t\t\t: %d.%d\n", major, minor);
	printf("> No. MultiProcesadores\t\t\t: %d\n", SM);
	printf("> No. Nucleos CUDA (%dx%d)\t\t: %d\n", cudaCores, SM, cudaCores * SM);
	printf("> No. maximo de Hilos (por bloque)\t: %d\n", maxThreads);
	printf("> Memoria Global (total)\t\t: %u MiB\n", deviceProp.totalGlobalMem / (1024 * 1024));
}

__global__ void ordenarVector(int* vector, int* resultado, int elementos){
	// Calculamos la posicion de cada hilo
	int x = threadIdx.x;
	// Rango del elemento.
	int count = 0;

	// Recorremos el vector
	for (int i = 0; i < elementos; i++){
		//Si el elemento del hilo acutal es mayor que el recorrido o son iguales, aumentamos el rango
		if (vector[x] > vector[i] || vector[x] == vector[i] && x > i){
			count++;
		}
	}
	// Asignamos el elemento al vector ordenado
	resultado[count] = vector[x];
}

// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)

// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{

	//declaraciones
	int* dev_vector, * dev_resultado;
	int* hst_vector, * hst_resultado;
	int elementos = 0;
	int bloques = 1, hilos;


	// Busqueda de dispositivos
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	// Muestra información del dispositivo seleccionado o un error si no lo hay
	if (deviceCount == 0)
	{
		// Muestra un error por no encontrar dispositivos compatibles con CUDA
		printf("!!!!!No se han encontrado dispositivos CUDA!!!!!\n");
		printf("<pulsa [INTRO] para finalizar>");
		getchar();

		// Termina el programa
		return 1;
	}
	else
	{
		// Toma el ID del dispositivo seleccionado
		int dispositivoSeleccionado;
		cudaGetDevice(&dispositivoSeleccionado);

		// Muestra los datos del dispositivo
		propiedades_Device(dispositivoSeleccionado);

	}

	//Pedimos al usuario el numero de elementos a ordenar
	printf("***************************************************\n");
	printf("  Introduce cuantos elementos se desean ordenar: ");
	scanf("%d", &elementos);

	//Calculamos los hilos  bloques en base al numero de elementos dados
	//En mi caso 32 hilos por 32 bloques
	hilos = elementos % 32;
	bloques += elementos / 32;
	
	//En caso de superar el limite o ser menor que 0, se pide otro valor
	while (elementos < 0 || elementos > 1024) {
		printf("Valor incorrecto");
		printf("Introduce cuantos elementos se desean ordenar, maximo %d: ", 1024);
		scanf("%d", &elementos);
	}

	
	//reserva en el host
	hst_vector = (int*)malloc(elementos * sizeof(int));
	hst_resultado = (int*)malloc(elementos * sizeof(int));

	//reserva en el device
	cudaMalloc((void**)&dev_vector, elementos * sizeof(int));
	cudaMalloc((void**)&dev_resultado, elementos * sizeof(int));

	//inicializar array
	for (int i = 0; i < elementos; i++) {
		hst_vector[i] = (int)rand() % 30;
	}
	

	//copia de datos al device
	cudaMemcpy(dev_vector, hst_vector, elementos * sizeof(float), cudaMemcpyHostToDevice);

	// Lanzamos un kernel bidimensional de un hilo
	dim3 hilosB(NBLOQUES*NHILOS);

	printf("> Kernel con %d bloques de %d hilos (%d hilos)\n", bloques, NHILOS, elementos);

	// Declaración del evento que calcula el tiempo de ejecución
	cudaEvent_t start;
	cudaEvent_t stop;

	// Creacion del evento
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Captura de la marca de tiempo de inicio
	cudaEventRecord(start, 0);

	// Generamos el bitmap de grises
	ordenarVector<<<1, hilosB >>> (dev_vector, dev_resultado, elementos);

	// Captura el final de la marca de tiempo
	cudaEventRecord(stop, 0);

	// Sincronizacion GPU-CPU
	cudaEventSynchronize(stop);

	// Calculo del tiempo en milisegundos
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	// Impresion de resultados
	printf("> Tiempo de ejecucion: %f ms\n", elapsedTime);

	// Finalización del evento
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copiamos los datos ordenados del device al host
	cudaMemcpy(hst_resultado, dev_resultado, elementos * sizeof(float), cudaMemcpyDeviceToHost);

	//Mostramos el vector inicial
	printf("> VECTOR INICIAL:\n");
	for (int i = 0; i < elementos; i++) {
		printf("%d ", hst_vector[i]);
	}

	//Mostramos el vector obtenido
	printf("\n> VECTOR ORDENADO:\n");
	for (int i = 0; i < elementos; i++){
		printf("%d ", hst_resultado[i]);
	}

	// Salida
	printf("\n...pulsa [ESC] para finalizar...");

	return 0;
}