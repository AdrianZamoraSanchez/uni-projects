/*
* ARQUITECTURA DE COMPUTADORES
* Hecho por: Adrián Zamora Sánchez
* Ejercicio: Entregable 1 de CUDA
* Descripción: Genera una imagen de un tablero de ajedrez
*/

// INCLUDES
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <device_launch_parameters.h>
#include "gpu_bitmap.h"

// Constantes del compilador
#define ANCHO 512 // Dimension horizontal
#define ALTO 512 // Dimension vertical

// Función que muestra los datos del ID del dispositivo que se le pasa como párametro
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

// Función kernel, llamada desde el host y ejecutada en el device
__global__ void kernel(unsigned char* imagen)
{
	// Kernel bidimensional multibloque
	// coordenada horizontal de cada hilo
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	
	// coordenada vertical de cada hilo
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	// índice lineal para cada hilo
	int myID = x + y * blockDim.x * gridDim.x;
	
	// cada hilo obtiene la posicion de su pixel
	int miPixel = myID * 4;

	/* 
	 Determina los indices de cada fila y columna,
	 para ello divide el numero de fila/columna entre los bloques que representan las celdas del tablero,
	 a su vez se obitne el numero de celda del tablero divididendo en 8 (que es el numero de filas y columnas del tablero 8x8)
	 el número de pixeles de la imagen que se generará.
	*/
	int numeroFila = (x / (ANCHO / 8));
	int numeroColumna = (y / (ALTO / 8));

	// Comprueba si la cuadricula es par, sumando el indice de fila + columna
	bool esCuadriculaPar = (numeroFila + numeroColumna) % 2 == 0;

	/*
	Para entender este método podemos imaginar el tablero de ajedrez como una matriz, la cual tiene 64 numeros,
	si comenzamos contando por el 1 hasta el 65, comienza en numero impar, luego se asigna color blanco, luego 2 que es
	par, le corresponde negro, y así sucesivamente para rellenar los ANCHO*ALTO pixeles separados en 8 bloques (celdas del tablero)
	en cada uno de los ejes x e y (filas y columnas del tablero).
	
	Para ilustrarlo:

	El tablero comienza y acaba:
	{ 0 PAR (blanco),  1 IMPAR (negro) ..... 63 IMPAR (negro) ,65 PAR (blanco)}
	*/

	// Colorea cada pixel, de blanco si es impar y de negro si es impar
	if (esCuadriculaPar == 0)
	{
		// Color negro
		imagen[miPixel + 0] = 0;   // canal R
		imagen[miPixel + 1] = 0;   // canal G
		imagen[miPixel + 2] = 0;   // canal B
		imagen[miPixel + 3] = 0;   // canal alfa
	}
	else
	{
		// Color blanco
		imagen[miPixel + 0] = 255; // canal R
		imagen[miPixel + 1] = 255; // canal G
		imagen[miPixel + 2] = 255; // canal B
		imagen[miPixel + 3] = 0;   // canal alfa
	}
}

// Rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// Toma el total de dispositivos compatibles con CUDA
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

	// Declaracion del bitmap:
	// Inicializacion de la estructura RenderGPU con el ancho y alto definido
	RenderGPU foto(ANCHO, ALTO);

	// Se genera el bitmap con el que se va a trabajar
	size_t bmp_size = foto.image_size();

	// Asignación y reserva de la memoria en el host (framebuffer)
	unsigned char* host_bitmap = foto.get_ptr();

	// Reserva en el device
	unsigned char* dev_bitmap;
	cudaMalloc((void**)&dev_bitmap, bmp_size);

	// Lanzamos un kernel bidimensional con bloques de 256 hilos (16x16)
	dim3 hilosB(16, 16);

	// Calculamos el numero de bloques necesario (un hilo por cada pixel)
	dim3 Nbloques(ANCHO / 16, ALTO / 16);

	// Kernel que genera el bitmap
	kernel <<<Nbloques, hilosB>>> (dev_bitmap);

	// Copiamos los datos desde la GPU hasta el framebuffer para visualizarlos
	cudaMemcpy(host_bitmap, dev_bitmap, bmp_size, cudaMemcpyDeviceToHost);
	
	// Visualizacion del bitmap y evita que se termine el programa hasta que el usuario lo cierre
	foto.display_and_exit();

	return 0;
}