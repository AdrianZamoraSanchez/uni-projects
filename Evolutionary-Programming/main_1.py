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

def inicializar_genotipo():
    """
    Genera los genotipos iniciales, para ello coloca en cada caché
    una serie de videos sin exceder su capacidad
    """
    # Variable que contiene las asignaciones video-cache
    individuo = [[] for _ in range(data["num_caches"])]
    
    # Se recorren todas las caches del individuo
    for cacheID in range(data["num_caches"]):
        # Variables que controlan la capacidad de la cache
        full = False
        cacheSize = 0

        # Cada cacheID tiene su propia lista previamente inicializada
        individuo[cacheID] = []

        # Se añaden videos mientras no este lleno ni se alcance el 80% de uso de la caché
        while not full and cacheSize < (data["cache_capacity"]*0.8):
            # Extra un video aleatorio
            videoID = random.randint(0, data["num_videos"]-1)

            # Se asigna el video a la cache que corresponde
            individuo[cacheID].append(videoID)

            # Se suma al uso de la cache esa asignación
            cacheSize += data["video_sizes"][videoID]

            # Se comprueba si la caché esta llena
            if cacheSize > data["cache_capacity"]:
                full = True

    # Devuelve al individuo
    return individuo

def evalSol(ind):
    """
    Evalúa la mejora de latencia obtenida al usar cachés frente al datacenter.
    """
    total_latencia = 0
    latencyToDataCenter = 0

    # Convertir las cachés del individuo a conjuntos
    cache_sets = [set(cache) for cache in ind]

    for video_id, endpoint_id, num_requests in data["requests"]:
        endpoint_data = data["endpoints"][endpoint_id]
        datacenter_latency = endpoint_data["datacenter_latency"]
        cache_connections = endpoint_data["cache_connections"]

        # Suma latencia directa al datacenter
        latencyToDataCenter += datacenter_latency * num_requests

        # Calcula la latencia mínima con cachés
        min_latency = datacenter_latency
        for cache_id, cache_latency in cache_connections.items():
            if video_id in cache_sets[cache_id]:
                min_latency = min(min_latency, cache_latency)

        # Suma la latencia mínima para esta solicitud
        total_latencia += min_latency * num_requests

    # Diferencia entre latencia sin caché y con caché
    return (latencyToDataCenter - total_latencia,)

def stopMethod(logbook, threshold, patience):
    """
    Comprueba si el algoritmo ha convergido en función de la mejora porcentual.
    - threshold: mejora mínima requerida para seguir buscando.
    - patience: número de generaciones que no cumplen el threshold.
    """
    # Se emplea como medida de rendimiento el fitness máximo
    max_fitness = logbook.select("max_fitness")
    
    # Si hay menos de "patience" generaciones, no se puede verificar convergencia
    if len(max_fitness) < patience:
        return False

    # Se comprueba la mejora conseguida
    for i in range(0, patience):
        # Calcula la mejora porcentual
        improvement = (max_fitness[-i] - max_fitness[-i-1]) / max_fitness[-i-1]

        if improvement >= threshold:
            return False  # Continua el algoritmo

    return True # Termina el algoritmo

def mutarAsignacion(individuo, indpb):
    """
    Esta función muta el individuo, eligiendo aleatoriamente un video y asignándolo
    a una caché de manera que no se sobrepase la capacidad máxima de la caché.

    No ha dado muy buenos resultados
    """
    for i in range(len(individuo)):
        if random.random() < indpb:
            # Elegir una caché aleatoria y un video aleatorio
            mutCacheID = random.randint(0, data["num_caches"] - 1)
            mutVideoID = random.randint(0, data["num_videos"] - 1)

            # Verificar la capacidad de la caché que se quiere mutar
            cacheSize = sum(data["video_sizes"][videoID] for videoID in individuo[mutCacheID])

            # Se comprueba que la caché tiene suficiente capacidad
            if cacheSize + data["video_sizes"][mutVideoID] <= data["cache_capacity"]:
                # Agregar el video a la caché seleccionada
                individuo[mutCacheID].append(mutVideoID)
            else:
                # Si la caché no tiene suficiente capacidad, intentar mover a otra caché
                pass

    return individuo,  # Debe retornar una tupla, ya que DEAP lo requiere

# Lectura del archivo especificado
filename = "kittens.in"
data = read_input(filename)

if __name__ == "__main__":
    # Configuración de DEAP para maximizar
    toolbox = base.Toolbox()
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Registrar funciones en DEAP
    toolbox.register("individual", tools.initIterate, creator.Individual, inicializar_genotipo)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalSol)

    # Proceso de selección
    toolbox.register("select", tools.selTournament, tournsize=3)
    #toolbox.register("select", tools.selBest)
    #toolbox.register("select", tools.selDoubleTournament,  fitness_size=4, parsimony_size=2, fitness_first=True)
    #toolbox.register("select", tools.selRandom)
    #toolbox.register("select", tools.selStochasticUniversalSampling)

    # Proceso de cruce
    #toolbox.register("mate", tools.cxOnePoint)
    #toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    
    # Proceso de mutación
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.02)
    #toolbox.register("mutate",mutarAsignacion, indpb=0.02) # Metodo personalizado

    # Cantidad de individuos que tiene cada generación
    population = toolbox.population(n=40)

    # Valores de probabilidad
    probCruce = 0.5
    probMut = 0.08

    # Número de generaciones
    numGen = 14

    # Se configura que estadísticas se quieren analizar 
    stats = tools.Statistics(lambda ind: ind.fitness.values) 
    stats.register("avg_fitness", np.mean)
    stats.register("max_fitness", np.max)
    stats.register("min_fitness", np.min)

    # Configurar el toolbox para usar paralelismo
    pool = Pool()  # Por defecto, utiliza todos los núcleos disponibles
    toolbox.register("map", pool.map)

    # Inicia el cronómetro
    start = time.time()

    # Variable que recoge los resultados
    logbook = tools.Logbook()
    logbook.header = ["gen", "avg_fitness", "max_fitness", "min_fitness"]

    # Bucle que ejecuta el algoritmo para cada genración
    for gen in range(numGen):
        # Se ejecuta una única generación
        population, _ = algorithms.eaSimple(
            population, toolbox,
            cxpb=probCruce, mutpb=probMut,
            ngen=1, verbose=False, stats=stats
        )

        # Seleccionar a los elites
        nElites = 3
        elites = tools.selBest(population, nElites)

        # Añadir los élites a la población de la próxima generación
        population[-nElites:] = elites

        # Guardar estadísticas de la generación actual
        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        print(logbook.stream)

        # Verificar si ha habido convergencia
        if stopMethod(logbook, threshold=5e-10, patience=20):
            print(f"Convergencia alcanzada en la generación {gen}")
            break


    # Detiene el cronómetro
    end = time.time()

    # Cerrar el pool al finalizar
    pool.close()
    pool.join()

    # Se recuperan los datos desde el log
    gen = logbook.select("gen")
    avgs = logbook.select("avg_fitness")
    maxim = logbook.select("max_fitness")
    minim = logbook.select("min_fitness")

    # Se establece una figura para dibujar
    fig = plt.figure()
        
    # Se representa la media del valor de fitness por cada generación
    ax1 = plt.gca()
    ax2 = plt.gca()
    ax3 = plt.gca()
    line1 = ax1.plot(gen, avgs, "g-", label="Average Fitness")
    line2 = ax2.plot(gen, maxim, "r-", label="Max Fitness")
    line3 = ax3.plot(gen, minim, "b-", label="Min Fitness") 
    ax1.set_xlabel("Generation", color="black")
    ax1.set_ylabel("Fitness", color="black")

    # Mostrar la gráfica
    plt.legend()
    plt.show()

    # Se seleccioan el mejor individuo
    ind = tools.selBest(population,1)[0]

    # Calcula el tiempo transcurrido
    processTime = end - start
    print(f"Tiempo de procesamiento: {processTime:.6f} segundos")

    print(f"La mejor solucion encontrada es el fenotipo: {evalSol(ind)[0]}")