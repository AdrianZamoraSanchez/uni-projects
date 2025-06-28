# Librerías necesarias
import random
import numpy as np

import matplotlib.pyplot as plt

from deap import base, creator
from deap import tools
from deap import algorithms

from multiprocessing import Pool

import time

def read_input(filename):
    """
    Función que lee el archivo .in, extrae y devuelve los datos del problema
    """
    with open(filename, 'r') as file:
        # Leer la primera línea de entrada con los parámetros principales
        first_line = file.readline().strip().split()
        num_videos = int(first_line[0])
        num_endpoints = int(first_line[1])
        num_requests = int(first_line[2])
        num_caches = int(first_line[3])
        cache_capacity = int(first_line[4])

        # Leer los tamaños de los videos
        video_sizes = list(map(int, file.readline().strip().split()))

        # Leer la información de cada endpoint y almacenarla como un diccionario
        endpoints = {}
        for endpoint_id in range(num_endpoints):
            # Leer latencia al centro de datos y cantidad de cachés conectados
            endpoint_info = file.readline().strip().split()
            datacenter_latency = int(endpoint_info[0])
            num_connected_caches = int(endpoint_info[1])
            
            # Crear un diccionario para las conexiones de cachés
            cache_connections = {}
            for _ in range(num_connected_caches):
                cache_id, cache_latency = map(int, file.readline().strip().split())
                cache_connections[cache_id] = cache_latency
            
            # Almacenar el endpoint como un diccionario
            endpoints[endpoint_id] = {
                "datacenter_latency": datacenter_latency,
                "cache_connections": cache_connections
            }

        # Leer las solicitudes de video y almacenarlas como una lista de tuplas
        requests = []
        for _ in range(num_requests):
            video_id, endpoint_id, num_requests = map(int, file.readline().strip().split())
            requests.append((video_id, endpoint_id, num_requests))

    # Devolver los datos organizados
    return {
        "num_videos": num_videos,
        "num_endpoints": num_endpoints,
        "num_requests": num_requests,
        "num_caches": num_caches,
        "cache_capacity": cache_capacity,
        "video_sizes": video_sizes,
        "endpoints": endpoints,
        "requests": requests
    }

def calc_cache_usage(ind):
    # Convertir las cachés del individuo a conjuntos
    cache_sets = [set(cache) for cache in ind]

    # Variable que almacena el uso del cache
    total_cache_usage = 0

    for video_id, endpoint_id, num_requests in data["requests"]:
        endpoint_data = data["endpoints"][endpoint_id]
        cache_connections = endpoint_data["cache_connections"]

        # Acumula el uso de la capacidad
        for cache_id in cache_connections:
            if video_id in cache_sets[cache_id]:
                total_cache_usage += data["video_sizes"][video_id]

    return total_cache_usage

def inicializar_genotipo():
    """
    Genera los genotipos iniciales, colocando videos aleatorios en cada caché
    esta asignación puede exceder la capacidad de las cachés
    """
    # Se define el individuo
    individuo = [[] for _ in range(data["num_caches"])]

    # Lista de videos disponibles
    videos_disponibles = list(range(data["num_videos"]))

    # Mezclar los videos para que cada individuo sea aleatorio
    # Este método de asig
    random.shuffle(videos_disponibles)  

    # Se asignan como mucho num videos / 2 para evitar que el algoritmo
    # tarde demasiado en encontrar individuos válidos por exceso de asignaciones
    videos = round(data["num_videos"]/2)

    for cacheID in range(data["num_caches"]):
        # Número de videos a añadir
        max_videos_por_cache = random.randint(1, videos)

        while len(videos_disponibles) > 0 and len(individuo[cacheID]) < max_videos_por_cache:
            # Seleccionar el último video disponible
            videoID = videos_disponibles.pop() 
            individuo[cacheID].append(videoID)

    return individuo

def eval_sol(ind):
    """
    Evalúa la mejora de latencia y el uso de la capacidad de las cachés.
    """
    total_latencia = 0
    latency_data_center = 0

    # Convertir las cachés del individuo a conjuntos
    cache_sets = [set(cache) for cache in ind]

    # Inicializar la capacidad usada por caché
    cache_usage = [0] * data["num_caches"]

    for video_id, endpoint_id, num_requests in data["requests"]:
        endpoint_data = data["endpoints"][endpoint_id]
        datacenter_latency = endpoint_data["datacenter_latency"]
        cache_connections = endpoint_data["cache_connections"]

        # Suma latencia directa al datacenter
        latency_data_center += datacenter_latency * num_requests

        # Calcula la latencia mínima con cachés
        min_latency = datacenter_latency
        for cache_id, cache_latency in cache_connections.items():
            if video_id in cache_sets[cache_id]:
                min_latency = min(min_latency, cache_latency)

        # Suma la latencia mínima para esta solicitud
        total_latencia += min_latency * num_requests

        # Acumula el uso de la capacidad
        for cache_id in cache_connections:
            if video_id in cache_sets[cache_id]:
                cache_usage[cache_id] += data["video_sizes"][video_id]

    # Calculo del uso de este individuo
    total_cache_usage = 0
    for cache in cache_usage:
        total_cache_usage += cache
    
    # Calcular la capacidad total
    total_capacity = data["cache_capacity"] * data["num_caches"]

    # Espacio libre disponible que se quiere maximizar 
    # pues to que el espacio libre capacidad - espacio usado crece cuanto
    # más se ahorra espacio utilizado
    free_capacity = (total_capacity - total_cache_usage)
    if(free_capacity < 0):
        free_capacity = 0

    # Calcula la mejora de latencia, se quiere maximizar el 
    # ahorro de tiempo en las requests, es decir minimizar 
    # lo bueno que es el uso del data center frente al uso de las caches
    latency_gained = latency_data_center - total_latencia

    # Devolver la latencia mejorada y la capacidad ajustada
    return (0.6*latency_gained, 0.4*free_capacity)

def feasible(ind):
    """
    Función de factibilidad para el individuo.
    Devuelve True si el individuo es válido (ninguna caché excede la capacidad)
    """
    # Inicializar la capacidad usada por caché
    cache_usage = [0] * data["num_caches"]

    # Convertir las cachés del individuo a conjuntos
    cache_sets = [set(cache) for cache in ind]

    for video_id, endpoint_id, num_requests in data["requests"]:
        endpoint_data = data["endpoints"][endpoint_id]
        cache_connections = endpoint_data["cache_connections"]

        # Acumula el uso de la capacidad
        for cache_id in cache_connections:
            if video_id in cache_sets[cache_id]:
                cache_usage[cache_id] += data["video_sizes"][video_id]

    # Comprueba si el uso de alguna cache sobrepasa la capacidad
    for usage in cache_usage:
        if usage > data["cache_capacity"]:
            return False
    
    # El individuo es válido
    return True


def distance(ind):
    """
    Función de distancia respecto de los resultados válidos
    Devuelve cuánto excede la capacidad el individuo
    """
    # Inicializar la capacidad usada por caché
    cache_usage = [0] * data["num_caches"]

    # Convertir las asignaciones del individuo a conjuntos
    cache_sets = [set(cache) for cache in ind]

    for video_id, endpoint_id, _ in data["requests"]:
        endpoint_data = data["endpoints"][endpoint_id]
        cache_connections = endpoint_data["cache_connections"]

        # Acumula el uso de la capacidad
        for cache_id in cache_connections:
            if video_id in cache_sets[cache_id]:
                cache_usage[cache_id] += data["video_sizes"][video_id]

    # Calcular el exceso de capacidad total
    total_excess = sum(max(0, usage - data["cache_capacity"]) for usage in cache_usage)
    return total_excess

# Lectura del archivo especificado
filename = "me_at_the_zoo.in"
data = read_input(filename)

if __name__ == "__main__":
    # Configuración de DEAP para maximizar
    toolbox = base.Toolbox()
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    # Registrar funciones en DEAP
    toolbox.register("individual", tools.initIterate, creator.Individual, inicializar_genotipo)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_sol)
    
    # Comprobación de la validez y distancia de los individuos invalidos respecto de la región válida
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 1.0, distance))
    
    # Proceso de selección
    toolbox.register("select", tools.selSPEA2)

    # Proceso de cruce
    toolbox.register("mate", tools.cxUniform, indpb=0.5)

    # Proceso de mutación
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.02)

    # Cantidad de individuos que tiene cada generación
    population = toolbox.population(n=50)

    # Valores de probabilidad
    probCruce = 0.5
    probMut = 0.08

    # Número de generaciones
    numGen = 100

    # Se configura que estadísticas de fitness que se quieren analizar 
    stats  = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg_fitness", np.mean)
    stats.register("max_fitness", np.max)
    stats.register("min_fitness", np.min)

    # Inicia el cronómetro
    start = time.time()

    # Variable que recoge los resultados
    logbook = tools.Logbook()
    logbook.header = ["gen", "avg_fitness", "max_fitness", "min_fitness", "avg_capacity_usage"]

    # Bucle que ejecuta el algoritmo para cada generación
    for gen in range(numGen):
        # Se ejecuta una única generación
        population, _ = algorithms.eaSimple(
            population, toolbox,
            cxpb=probCruce, mutpb=probMut,
            ngen=1, verbose=False, stats=stats
        )

        # Calcular el uso de capacidad promedio en la población
        # hacer bien la media de % de uso porque sale muy alta, quizas porque es la media de muchos ind
        avg_capacity_usage = np.mean([calc_cache_usage(ind) for ind in population])

        # Guardar estadísticas de la generación actual
        record = stats.compile(population)
        record["avg_capacity_usage"] = (avg_capacity_usage/(data["cache_capacity"]*data["num_caches"]))*100  # Añadir al registro

        logbook.record(gen=gen, **record)  # Registrar en el logbook
        print(logbook.stream)

        # Seleccionar a los elites
        nElites = 3
        elites = tools.selBest(population, nElites)

        # Añadir los élites a la población de la próxima generación
        population[-nElites:] = elites

        # Guardar estadísticas de la generación actual
        record = stats.compile(population)

    # Detiene el cronómetro
    end = time.time()

    # Se recuperan los datos desde el log
    gen = logbook.select("gen")
    avgs = logbook.select("avg_fitness")
    maxim = logbook.select("max_fitness")
    minim = logbook.select("min_fitness")
    capacities = logbook.select("avg_capacity_usage")

    # Crear una figura para la visualización
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Graficar la media de fitness y la capacidad promedio
    ax1.plot(gen, avgs, "g-", label="Average fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.legend()

    # Graficar la media de uso de memoria de las cachés (en %)
    ax2.plot(gen, capacities, "b-", label="Capacity usage")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Capacity usage (%)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Se selecciona el mejor individuo
    best = tools.selBest(population, 1)[0]

    # Calcular el uso de la capacidad para el mejor individuo
    bestCapacityUsage = (calc_cache_usage(best) / (data["cache_capacity"]*data["num_caches"]))*100

    # Calcula el tiempo transcurrido
    processTime = end - start

    # Se muestran los datos más relevantes sobre el resultado obtenido
    print(f"Tiempo de procesamiento: {processTime:.6f} segundos")
    print(f"La mejor solución encontrada es el fenotipo: {eval_sol(best)[0]}")
    print(f"El mejor individuo utiliza una capacidad del {bestCapacityUsage:.6f}%")