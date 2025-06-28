/*
* ARQUITECTURA DE COMPUTADORES
* Hecho por: Adrián Zamora Sánchez y Adrián Alcalde Alzaga
* Ejercicio: Entregable 5 de CUDA
* Descripción: Ordenar de menor a mayor un vector de N elementos aleatorios 
               utilizando el método de ordenación por rango en GPU y compara su tiempo
			   de ordenación con la misma ordenación hecha en una CPU. Finalmente
			   muestra un resumen del rendimiento obtenido con diferentes configuraciones.
*/

// includes
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#ifdef __linux__
#include <sys/time.h>
typedef struct timeval event;
#else
#include <windows.h>
typedef LARGE_INTEGER event;
#endif

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
	printf("> Memoria Global (total)\t\t: %zu MiB\n", deviceProp.totalGlobalMem / (1024 * 1024));
}

// Función que genera una nueva marca de tiempo
__host__ void setEvent(event* ev)
{
#ifdef __linux__
	gettimeofday(ev, NULL);
#else
	QueryPerformanceCounter(ev);
#endif
}

// Función que devuelve la diferencia de tiempo (en ms) entre dos eventos
__host__ double eventDiff(event* first, event* last)
{
#ifdef __linux__
	return
		((double)(last->tv_sec + (double)last->tv_usec / 1000000) -
			(double)(first->tv_sec + (double)first->tv_usec / 1000000)) * 1000.0;
#else
	event freq;
	QueryPerformanceFrequency(&freq);
	return ((double)(last->QuadPart - first->QuadPart) / (double)freq.QuadPart) * 1000.0;
#endif
}

// Función de reducción para calcular la posición de cada elemento en el vector ordenado
__device__ int reduce(int* vector, int x, int elementos) {
	int count = 0;

	/*	
		Compara el elemento del hilo actual con los demás elementos
		y cuenta cuántos elementos son menores o tienen el mismo valor
		pero están en posiciones anteriores en el vector original
	*/
	for (int i = 0; i < elementos; i++) {
		if (vector[x] > vector[i] || (vector[x] == vector[i] && x > i)) {
			count++;
		}
	}

	return count;
}

// Funcion llamada desde el host y ejecutada en el device (kernel) responsable de ordenar el vector en la GPU
__global__ void ordenarVector(int* vector, int* resultado, int elementos) {
	// Número del hilo
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	// Cada hilo calcula su posición en el vector ordenado utilizando la función reduce
	int count = reduce(vector, x, elementos);

	// Asigna el elemento al vector ordenado en la posición calculada
	resultado[count] = vector[x];
}

// Función que ordena los vectores ejecutada por la CPU (host)
__host__ void ordenarVectorCPU(int* vector, int* resultado, int elementos) {

	// Se recorre el vector
	for (int i = 0; i < elementos; i++) {
		// Se toma el rango del elemento a ordenar
		int count = 0;

		// Se recorre el vector nuevamente para comparar con otros elementos
		for (int j = 0; j < elementos; j++) {
			// Si el elemento del índice actual es mayor que el recorrido se aumenta el rango
			if (vector[i] > vector[j] || (vector[i] == vector[j] && i > j)) {
				count++;
			}
		}
		// Se asigna el elemento al vector ordenado
		resultado[count] = vector[i];
	}
}

/* 
	Estructura que guarda para cada ejecución los valores del tiempo empleado por
	GPU, CPU, la ganancia entre ellas y el número de hilos con el que se ha ejecutado
*/
typedef struct {
	float elapsedTimeGPU;
	double elapsedTimeCPU;
	double ganancia;
	int nHilos;
} Resultados;

// Rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// Arrays del programa
	int *dev_vector, *dev_resultado; // Arrays del device
	int *hst_vector, *hst_resultado_dev, *hst_resultado_cpu; // Arrays en el host

	// Variables de control
	int elementos = 0, mostrarVectores;
	boolean acabado = false; 
	int iteracion = 0; 

	double elapsedTimeCPU; // Variable para el tiempo de CPU
	int NHilosTotales=32, NHilos=1024; // Constantes sobre los hilos
	Resultados resultados[10]; // Array de resultados

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

	// Se solicita al usuario el número de elementos a ordenar
	printf("***************************************************\n");
	printf("  Introduce el tamano del vector: ");
	scanf("%d", &elementos);
	
	// En caso de superar el limite o ser menor que 0, se pide otro valor
	while (elementos < 0 || elementos > 4096) {
		printf("Valor incorrecto");
		printf("Introduce cuantos elementos se desean ordenar, maximo %d: ", 1024);
		scanf("%d", &elementos);
	}

	// Se solicita opción para mostrar o no los resultados
	printf(" Mostrar los vectores por pantalla [NO:0, SI:1]?: ");
	scanf("%d", &mostrarVectores);

	// Se ejecutan varias pruebas de rendimiento
	while (acabado == false) {

		// Se calcula el número de Bloques
		int NBloques = NHilosTotales / NHilos;

		// Como mínimo se debe utilizar 1
		if (NBloques == 0) {
			NBloques = 1;
		}

		// Reserva en el host
		hst_vector = (int*)malloc(elementos * sizeof(int));
		hst_resultado_dev = (int*)malloc(elementos * sizeof(int));
		hst_resultado_cpu = (int*)malloc(elementos * sizeof(int));

		// Reserva en el device
		cudaMalloc((void**)&dev_vector, elementos * sizeof(int));
		cudaMalloc((void**)&dev_resultado, elementos * sizeof(int));

		// Inicialización del array
		for (int i = 0; i < elementos; i++) {
			hst_vector[i] = (int)rand() % 30;
		}

		//copia de datos al device
		cudaMemcpy(dev_vector, hst_vector, elementos * sizeof(float), cudaMemcpyHostToDevice);

		// Lanzamos un kernel bidimensional
		dim3 hilosB(NBloques * NHilos);

		// Declaración del evento que calcula el tiempo de ejecución
		cudaEvent_t startGPU;
		cudaEvent_t stopGPU;

		// Creacion del evento GPU
		cudaEventCreate(&startGPU);
		cudaEventCreate(&stopGPU);

		// Captura de la marca de tiempo de inicio
		cudaEventRecord(startGPU, 0);

		// Ordenamos el Vector
		ordenarVector <<<NBloques, NHilos >>> (dev_vector, dev_resultado, elementos);

		// Captura el final de la marca de tiempo
		cudaEventRecord(stopGPU, 0);

		// Sincronizacion GPU-CPU
		cudaEventSynchronize(stopGPU);

		// Calculo del tiempo en milisegundos de GPU
		float elapsedTimeGPU;
		cudaEventElapsedTime(&elapsedTimeGPU, startGPU, stopGPU);

		// Finalización del evento
		cudaEventDestroy(startGPU);
		cudaEventDestroy(stopGPU);

		// Se copian los datos ordenados del device al host
		cudaMemcpy(hst_resultado_dev, dev_resultado, elementos * sizeof(float), cudaMemcpyDeviceToHost);

		// Creacion del evento CPU
		event startCPU, stopCPU;

		// Captura de la marca de tiempo de CPU
		setEvent(&startCPU);

		//Ordenamos el vector mediante la CPU
		ordenarVectorCPU(hst_vector, hst_resultado_cpu, elementos);

		// Captura el final de la marca de tiempo de la CPU
		setEvent(&stopCPU);

		// Calculo del tiempo en milisegundos de CPU
		elapsedTimeCPU = eventDiff(&startCPU, &stopCPU);

		// Impresion de resultados
		printf("> Numero de elementos a ordenar: [%d]\n", elementos);
		printf("> Lanzamiento: %d hilos por bloque y %d bloques.\n", NHilos, NBloques);
		printf("> TOTAL: %d hilos\n", NHilosTotales);
		printf("  EJECUCION GPU...\n");
		printf("> Tiempo de ejecucion: %f ms\n", elapsedTimeGPU);
		printf("  EJECUCION CPU...\n");
		printf("> Tiempo de ejecucion: %f ms\n", elapsedTimeCPU);
		printf("> Ganancia GPU/CPU: %f", elapsedTimeCPU / elapsedTimeGPU);
		printf("\n***************************************************\n");
		resultados[iteracion].elapsedTimeGPU = elapsedTimeGPU;
		resultados[iteracion].elapsedTimeCPU = elapsedTimeCPU;
		resultados[iteracion].ganancia = elapsedTimeCPU / elapsedTimeGPU;
		resultados[iteracion].nHilos = NHilosTotales;

		//Incrementamos las iteraciones
		iteracion++;

		if (NHilosTotales >= 4096) {
			// Con el máximo de hilos (4096) se termina la ejecución de kernels
			acabado = true;
		}

		// Se incrementa el número de hilos a la siguiente potencia de 2
		NHilosTotales *= 2;
	}
	
	// Se muestra un resumen de los resultados obtenidos
	printf("RESUMEN DE RESULTADOS");
	printf("\n***************************************************\n");
	for (int i = 0; i < iteracion; i++) {
		printf("> N = %d | Hilos = %d [GPU: %f ms] [CPU: %f ms] [Ganancia GPU/CPU = %f]\n",
			elementos, resultados[i].nHilos, resultados[i].elapsedTimeGPU, resultados[i].elapsedTimeCPU,
			resultados[i].ganancia);
	}

	// Si el usuario lo ha solicitado se muestran los vectores
	if (mostrarVectores== 1) {
		// Vector inicial
		printf("> VECTOR INICIAL:\n");
		for (int i = 0; i < elementos; i++) {
			printf("%d ", hst_vector[i]);
		}

		// Vector resultados GPU
		printf("\n> VECTOR ORDENADO GPU:\n");
		for (int i = 0; i < elementos; i++) {
			printf("%d ", hst_resultado_dev[i]);
		}

		// Vector resultados CPU
		printf("\n> VECTOR ORDENADO CPU:\n");
		for (int i = 0; i < elementos; i++) {
			printf("%d ", hst_resultado_cpu[i]);
		}
	}

	// Salida y fin del programa
	printf("\n...pulsa [ESC] para finalizar...");
	return 0;
}