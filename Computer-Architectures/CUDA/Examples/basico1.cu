/*
* ARQUITECTURA DE COMPUTADORES
*
* EJEMPLO: "Basico 1 de CUDA"
* >> Copia un array de 4 números aleatorios desde la memoria del host
*    a la memoria de la GPU, luego la mueve a otra posición de la memoria
*	 de la GPU, finalmente la copia de vuelta en la memoria del host
*/

// INCLUDES
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// Define con el número de elementos que tienen los vectores con los que trabajaremos
#define ELEMENTOS 4

void propiedades_Device(int deviceID)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);
	// calculo del numero de cores (SP)
	int cudaCores = 0;
	int SM = deviceProp.multiProcessorCount;
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
	printf("***************************************************\n");
}

// Función MAIN del programa
int main() {

	// Busca los dispositivos CUDA
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

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
		printf("Se han encontrado <%d> dispositivos CUDA:\n", deviceCount);
		for (int id = 0; id < deviceCount; id++)
		{
			propiedades_Device(id);
		}
	}

	// Preparación de la generación de números aleatorios
	srand(time(nullptr));

	// Declaración y reserva de las variables para el host
	float *hst_A;
	hst_A = (float*)malloc(ELEMENTOS*sizeof(float));

	float *hst_B;
	hst_B = (float*)malloc(ELEMENTOS * sizeof(float));

	// Declaración de las variables para el device
	float *dev_A;
	float *dev_B;

	printf("Los numeros aleatorios iniciales en host (hst_A) son:\n");

	// Cargamos y visualizamos los datos del vector A del host
	for (int i = 0; i < ELEMENTOS; i++) {
		hst_A[i] = (float) rand() / RAND_MAX;
		printf("%.2f ", hst_A[i]);
	}

	// Reservamos espacio en memoria dentro del device 
	cudaMalloc((void**)&dev_A, ELEMENTOS * sizeof(float));

	// Copiamos a la memoria reservada anteriormente los valores del array inicial
	cudaMemcpy(dev_A, hst_A, ELEMENTOS * sizeof(float), cudaMemcpyHostToDevice);

	// Liberamos la memoria que acabamos de copiar
	free(hst_A);

	// Reservamos otro espacio en memoria para copiar
	cudaMalloc((void**)&dev_B, ELEMENTOS * sizeof(float));

	// Copiamos a este nuevo espacio los datos
	cudaMemcpy(dev_B, dev_A, ELEMENTOS * sizeof(float), cudaMemcpyDeviceToDevice);
	
	// Liberamos el primer bloque de memoria reservado en device
	cudaFree(dev_A);

	// Copiamos del device a la memoria principal que usaremos como resultado
	cudaMemcpy(hst_B, dev_B, ELEMENTOS * sizeof(float), cudaMemcpyDeviceToHost);

	// Liberamos el bloque de memoria reservado del device
	cudaFree(dev_B);

	// Mostramos el resultado
	printf("\nEl array resultado en host (hst_B): \n");
	for (int i = 0; i < ELEMENTOS; i++) {
		printf("%.2f ", hst_B[i]);
	}

	// Liberamos la última reserva de memoria
	free(hst_B);

	// Hacemos un log que incluya la finalización de la ejecución del programa
	time_t fecha;
	time(&fecha);
	printf("\n***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
}


