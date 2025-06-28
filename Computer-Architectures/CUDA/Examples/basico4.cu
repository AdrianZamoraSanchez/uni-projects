/*
* ARQUITECTURA DE COMPUTADORES
* Hecho por: Adrián Zamora Sánchez
* Ejercicio: "Básico 4 de CUDA"
* >> Generamos una matriz de tamaño F*C con valores aleatorios
* entre 1 y 9 en el host, pasamos esa matriz al device y lanzamos
* un kernel que cambia por 0 todos los valores en las columnas
* impares de la matriz, luego devuelve la matriz al host y este la imprime
*/

// INCLUDES
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <device_launch_parameters.h>

#define F 7
#define C 15

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

// Kernel del device donde se establecen las columnas impares de la matriz con 0
__global__ void generarMatrizDev(int* dev_matriz)
{	
	/* 
	Calcula el índice global para acceder a un elemento en una matriz simulada.
		-blockDim.y: Número de bloques en la dimensión de las columnas.
		-threadIdx.x: Número de fila del hilo dentro del bloque.
		-threadIdx.y: Número de columna del hilo dentro del bloque.
	 
		El índice de cada elemento se calcula como: bloque (blockDim.y) * fila (threadIdx.x) + columna (threadIdx.y) 
		puesto que realmente estaríamos apuntando a un índice lineal (la memoria principal) que simula dos dimensiones
	*/
	int idGlobal = threadIdx.y + threadIdx.x * blockDim.y;

	// Comparamos el índice de la columna
	if (threadIdx.y % 2 == 0) {
		// Es par, dejamos el valor
		dev_matriz[idGlobal] = dev_matriz[idGlobal]; 
	}
	else {
		// Es impar, ponemos un cero
		dev_matriz[idGlobal] = 0; 
	}
}

// Función donde el host genera la matriz de F*C valores aleatorios entre 1 y 9
__host__ void generarMatriz(int(*matriz)[C]) {
	
	printf("Matriz generada en host: \n");

	// Rellena e imprime cada valor de la matriz
	for (int i = 0; i < F; i++) {
		for (int j = 0; j < C; j++) {
			matriz[i][j] = 1 + rand() % 9; // Asigna un valor entre 1 y 9
			printf("%d ", matriz[i][j]);
		}
		printf("\n");
	}
}

// Función main
int main(int argc, char** argv)
{
	// Busca los dispositivos CUDA
	int deviceCount;

	// Generamos aleatoriedad en rand()
	srand(static_cast<unsigned>(time(nullptr)));

	// Tomamos la lista de dispositivos
	cudaGetDeviceCount(&deviceCount);

	// Comprueba que exista algún dispositivo
	if (deviceCount == 0)
	{
		// Sin no encuentra dispositivos CUDA deuvelve un error y termina el programa
		printf("No se han encontrado dispositivos CUDA!\n");
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

	// Definimos la matriz con la que trabaja el host
	int matriz[F][C];
	
	// Se define la matriz de dev
	int* dev_matriz;

	// Reservamos el tamaño completo de la matriz que será F*C*tamaño de un tipo int(4bytes)
	cudaMalloc((void**)&dev_matriz, F * C * sizeof(int));
	
	// Calculamos un número de bloques de 10 hilos aptos para la GPU dentro del eje X
	printf("Lanzado 1 bloque con: \neje x -> %d hilos\neje y -> %d hilos\n", C, F);

	// Llamamos a una función __host__ que da valores a la variable matriz
	generarMatriz(matriz);

	// Pasamos los valores de matriz a dev_matriz
	cudaMemcpy(dev_matriz, matriz, F * C * sizeof(int), cudaMemcpyHostToDevice);

	// Se lanzan (x = F) y (y = C) hilos en el bloque, los justos para realizar los calculos en esta matriz
	dim3 threadsPerBlock(F, C);
	
	// Se lanza un bloque con dos dimensiones
	dim3 numBlocks(1, 1);

	// Se llama al kernel bidimensional de F*C hilos
	generarMatrizDev <<<numBlocks, threadsPerBlock>>> (dev_matriz);

	// Copiamosl los resultados para su impresión por pantalla
	cudaMemcpy(matriz, dev_matriz, F * C * sizeof(int), cudaMemcpyDeviceToHost);

	// Imrpimimos la matriz resultante
	printf("Matriz device: \n");

	for (int i = 0; i < F; i++) {
		for (int j = 0; j < C; j++) {
			printf("%d ", matriz[i][j]);
		}
		printf("\n");
	}

	// Liberamos toda la memoria reservada dev
	cudaFree(dev_matriz);

	// Hacemos un log que incluya la finalización de la ejecución del programa
	time_t fecha;
	time(&fecha);
	printf("\n***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));

	// Termina el programa
	return 0;
}
