/*
* ARQUITECTURA DE COMPUTADORES
* Hecho por: Adrián Zamora Sánchez
* Ejercicio: "Básico 3 de CUDA"
* >> Generamos un array de N números aleatorios <0 >9 en host,
*    creamos otro array invirtiendo el anterior en dev, 
*    sumamos ambos en dev generando para ello tantos bloques de
*    un numero fijo de hilos que calculan la suma de arrays, finalmente 
*	 imprimimos el resultado desde host
*/

// INCLUDES
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <device_launch_parameters.h>

#define HILOS 10

// Función que muestra los datos del dispositivo
void propiedades_Device(int deviceID, int* maxThreads)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);

	// calculo del numero de cores (SP)
	int cudaCores = 0;
	int SM = deviceProp.multiProcessorCount;
	*maxThreads = deviceProp.maxThreadsPerBlock;
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

	printf("MAX Hilos por bloque: %d\n", deviceProp.maxThreadsPerBlock);
	printf("MAX BLOCK SIZE\n");
	printf(" [x -> %d]\n [y -> %d]\n [z -> %d]\n", deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("MAX GRID SIZE\n");
	printf(" [x -> %d]\n [y -> %d]\n [z -> %d]\n", deviceProp.maxGridSize[0],deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("***************************************************\n");
}

// Kernel del device donde se genera el vector inverso y se suman ambos
__global__ void sumarVectores(int* dev_vector1, int* dev_vector2, int* dev_vectorSuma, int N)
{
	int idGlobal = threadIdx.x + blockDim.x * blockIdx.x;

	dev_vector2[idGlobal] = dev_vector1[N - 1 - idGlobal];
	
	dev_vectorSuma[idGlobal] = dev_vector2[idGlobal] + dev_vector1[idGlobal];
}

// Función donde el host genera el vector de tamaño N con valores aleatorios
__host__ void generarVector(int* hst_vector1, int N) {
	;
	printf("Vector host: ");
	for (int i = 0; i < N; i++) {
		hst_vector1[i] = rand() % 10;
		printf("%d ", hst_vector1[i]);
	}
}

// Función main
int main(int argc, char** argv)
{
	// Variable de control para no sobrepasar el maximo de hilos de la GPU
	int maxThreads;

	// Busca los dispositivos CUDA
	int deviceCount;

	cudaGetDeviceCount(&deviceCount);
	//printf("Dimensiones de cada bloque: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);

	// Comprueba que exista algún dispositivo
	if (deviceCount == 0)
	{
		// Sin no encuentra dispositivos compatibles con CUDA deuvelve un error
		printf("No se han encontrado dispositivos CUDA!\n");
		return 1;
	}
	else
	{
		// Muestra los datos del dispositivo
		for (int id = 0; id < deviceCount; id++)
		{
			propiedades_Device(id, &maxThreads);
		}
	}

	// Generamos aleatoriedad en rand()
	srand(static_cast<unsigned>(time(nullptr)));

	// Declaramos los elementos del array (N) y el numero de bloques que vamos a usar
	int N, bloques;

	// Pedimos al usuario que introduza el valor de N
	printf("Introduce el numero de elementos\n");
	std::cin >> N; // Tomamos el input del usuario

	// Comprobamos que el usuario no haya creado un vector con más elementos que hilos
	if (maxThreads < N) {
		std::cout << "Error, se ha sobrepasado el maximo de hilos: " << maxThreads;
		return 0;
	}

	// Definimos los vectores con los que trabaja el host
	int* hst_vector1, * hst_vector2, * hst_vectorSuma;

	// Reservamos la memoria de los vectores host
	hst_vector1 = (int*)malloc(N * sizeof(int));
	hst_vector2 = (int*)malloc(N * sizeof(int));
	hst_vectorSuma = (int*)malloc(N * sizeof(int));

	// Definimos los vectores con los que trabaja dev
	int* dev_vector1, * dev_vector2, * dev_vectorSuma;

	// Reservamos la memoria e los vectores dev
	cudaMalloc(&dev_vector1, N * sizeof(int));
	cudaMalloc(&dev_vector2, N * sizeof(int));
	cudaMalloc(&dev_vectorSuma, N * sizeof(int));

	// Calculamos un número de bloques de 10 hilos aptos para la GPU dentro del eje X
	bloques = (int)ceil((double)N / 10);
	printf("Calculando con vectores de %d elementos\n", N);
	printf("Lanzado %d bloques de %d hilos (total: %d hilos)\n", bloques, HILOS, HILOS * bloques);

	// Llamamos a una función __host__ que davalors alatorios al vector1
	generarVector(hst_vector1, N);

	// Pasamos el vector 1 al dev
	cudaMemcpy(dev_vector1, hst_vector1, N * sizeof(int), cudaMemcpyHostToDevice);

	// Invertimos el vector y hacemos la suma en dev
	sumarVectores <<<bloques,HILOS>>> (dev_vector1, dev_vector2, dev_vectorSuma, N);

	// Copiamosl los resultados para su impresión por pantalla
	cudaMemcpy(hst_vector2, dev_vector2, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_vectorSuma, dev_vectorSuma, N * sizeof(int), cudaMemcpyDeviceToHost);

	// Imrpimimos el vector1 invertido (vector2)
	printf("\nVector device: ");
	for (int i = 0; i < N; i++) {
		printf("%d ", hst_vector2[i]);
	}

	// Imprimimos el valor de la suma (vectorSuma)
	printf("\nVector suma: ");
	for (int i = 0; i < N; i++) {
		printf("%d ", hst_vectorSuma[i]);
	}

	// Liberamos toda la memoria reservada en host y dev
	free(hst_vector1);
	free(hst_vector2);
	free(hst_vectorSuma);
	cudaFree(dev_vector1);
	cudaFree(dev_vector2);
	cudaFree(dev_vectorSuma);

	// Hacemos un log que incluya la finalización de la ejecución del programa
	time_t fecha;
	time(&fecha);
	printf("\n***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));

	// Termina el programa
	return 0;
}