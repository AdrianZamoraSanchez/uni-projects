/*
* ARQUITECTURA DE COMPUTADORES
* Hecho por: Adrián Zamora Sánchez y Adrián Alcalde Alzaga
* Ejercicio: Entregable 1 de CUDA
* Descripción: Generamos una matriz de tamaño F*C con valores aleatorios
* entre 1 y 9 iguales en cada fila en el host, pasamos esa matriz al device y lanzamos
* un kernel que cambia el orden de las filas, desplazandolas hacia abajo un fila. Además
* se implementa un evento de temporización que indica el tiempo empleado para esta tarea
* realizada en el device. 
*/

// INCLUDES
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <device_launch_parameters.h>

#define FILAS 7
#define COLUMNAS 25

// Función que muestra los datos del dispositivo
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
	printf("> CUDA Toolkit\t\t\t: %d.%d\n", rtV / 1000, (rtV % 1000) / 10);
	printf("> Arquitectura CUDA\t\t: %s\n", archName);
	printf("> Capacidad de Computo\t\t: %d.%d\n", major, minor);
	printf("> No. MultiProcesadores\t\t: %d\n", SM);
	printf("> No. Nucleos CUDA (%dx%d)\t: %d\n", cudaCores, SM, cudaCores * SM);
	printf("> Memoria Global (total)\t: %u MiB\n", deviceProp.totalGlobalMem / (1024 * 1024));

	printf("MAX Hilos por bloque: %d\n", maxThreads);
	printf("MAX BLOCK SIZE\n");
	printf(" [x -> %d]\n [y -> %d]\n [z -> %d]\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("MAX GRID SIZE\n");
	printf(" [x -> %d]\n [y -> %d]\n [z -> %d]\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("***************************************************\n");
}

// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void MatFinal(int* entrada, int* salida)
{
	// KERNEL BIDIMENSIONAL DE UN SOLO BLOQUE: (X,Y)
	// indice de columna: EJE x
	int columna = threadIdx.x;

	// indice de fila: EJE y
	int fila = threadIdx.y;

	// Índice global lineal
	int globalID = columna + fila * COLUMNAS;

	// Realizar el desplazamiento hacia abajo de las filas
	if (fila < FILAS - 1) {
		// Almacenar la fila actual en la matriz de salida
		salida[globalID + COLUMNAS] = entrada[globalID];
	}
	else {
		// Asignar la primera fila al final
		salida[columna] = entrada[globalID];
	}
}

// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// Busqueda de dispositivos
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	// Muestra información de los dispositivos encontrados
	if (deviceCount == 0)
	{
		printf("!!!!!No se han encontrado dispositivos CUDA!!!!!\n");
		printf("<pulsa [INTRO] para finalizar>");
		getchar();
		return 1;
	}
	else
	{
		// Muestra los datos de cada dispositivo encontrado
		for (int id = 0; id < deviceCount; id++)
		{
			propiedades_Device(id);
		}
	}
	
	// Declaraciones de las matrices
	int* hst_entrada, * hst_salida;
	int* dev_entrada, * dev_salida;
	
	// Reserva de las matrices en el host
	hst_entrada = (int*)malloc(FILAS * COLUMNAS * sizeof(int));
	hst_salida = (int*)malloc(FILAS * COLUMNAS * sizeof(int));
	
	// Reserva delas matrices en el device
	cudaMalloc((void**)&dev_entrada, FILAS * COLUMNAS * sizeof(int));
	cudaMalloc((void**)&dev_salida, FILAS * COLUMNAS * sizeof(int));
	
	// Incialización
	for (int i = 0; i < FILAS; i++)
	{
		int numRand = rand() % 9;

		for (int j = 0; j < COLUMNAS; j++) {
			hst_entrada[i * COLUMNAS + j] = numRand;
			hst_salida[i * COLUMNAS + j] = 0;
		}
	}
	
	// Dimensiones del kernel
	// 1 Bloque
	dim3 Nbloques(1);
	
	// Bloque bidimensional (x,y)
	// Eje x-> COLUMNAS
	// Eje y-> FILAS
	dim3 hilosB(COLUMNAS, FILAS);

	// Copia de datos hacia el device
	cudaMemcpy(dev_entrada, hst_entrada, FILAS * COLUMNAS * sizeof(int),cudaMemcpyHostToDevice);
	
	// Número de hilos
	printf("> KERNEL de 1 BLOQUE con %d HILOS:\n", COLUMNAS * FILAS);
	printf(" eje x -> %2d hilos\n eje y -> %2d hilos\n", COLUMNAS, FILAS);

	// Declaración del evento que calcula el tiempo de ejecución
	cudaEvent_t start;
	cudaEvent_t stop;

	// Creacion del evento
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Captura de la marca de tiempo de inicio
	cudaEventRecord(start, 0);
	
	// llamada al kernel
	MatFinal <<<Nbloques, hilosB>>> (dev_entrada, dev_salida);

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

	// Recogida de datos desde el device
	cudaMemcpy(hst_salida, dev_salida, FILAS * COLUMNAS * sizeof(int),cudaMemcpyDeviceToHost);

	// Impresión de los datos iniciales
	printf("> MATRIZ ORIGINAL:\n");
	
	for (int i = 0; i < FILAS; i++)
	{
		for (int j = 0; j < COLUMNAS; j++)
		{
			printf("%2d ", hst_entrada[j + i * COLUMNAS]);
		}
		printf("\n");
	}
	
	printf("\n");

	// Impresion de resultados obtenidos
	printf("> MATRIZ FINAL:\n"); 

	for (int i = 0; i < FILAS; i++)
	{
		for (int j = 0; j < COLUMNAS; j++)
		{
			printf("%2d ", hst_salida[j + i * COLUMNAS]);
		}
		printf("\n");
	}

	// Salida del programa donde se muestra la fecha y hora
	time_t fecha;
	time(&fecha);
	printf("***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	return 0;
}
