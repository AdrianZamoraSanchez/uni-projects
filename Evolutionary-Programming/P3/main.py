import numpy as np
import pandas as pd
import math
import random
from deap import base, creator, gp, tools, algorithms
import matplotlib.pyplot as plt
import operator
import time

def protected_div(left, right, limite=1e-3):
    """Evita casos de x/0 o valores muy cercanos a 0 que den resultados demasiado grandes"""
    if abs(right) < limite:
        return 1 # Valor seguro
    return left / right

def protected_sqrt(x):
    """Raiz protegida contra numeros negativos o incorrectos"""
    try:
        if x < 0:
            return 1 # Valor seguro
        return math.sqrt(x)
    except ValueError:
        return 1  # Valor seguro

def extract_data():
    """Extrae los datos y separa caracteristicas de datos de evaluación"""
    # Leer el archivo CSV
    csvFile = "body_fat.csv"
    df = pd.read_csv(csvFile)

    # Seleccionar todas las columnas excepto la segunda (que es la de resultados)
    data = df.drop(columns=df.columns[1])

    # Seleccionar solo la segunda columna (la de resultados)
    results = df.iloc[:, [1]]

    return data, results

def eval_sol(individuo):
    """
    Evalúa un individuo comparando las predicciones realizadas por la expresión que representa
    con los valores reales de los resultados (results) usando el MAE.
    
    Args:
        individuo: Un árbol de expresión que representa una posible solución.
    
    Returns:
        Una tupla con el MAE como único elemento, ya que estamos minimizando el error.
    """
    # Compilar el árbol de expresión del individuo
    func = toolbox.compile(expr=individuo)
    
    # Extraer datos y resultados
    data, results = extract_data()
    data = data.values  # Se convierte el DataFrame a matriz de np para manipulación más eficiente
    results = results.values.flatten()  # Convertir a un array unidimensional
    
    # Calcular las predicciones y los errores
    errores = []
    for x, real_value in zip(data, results):
        try:
            pred_value = func(*x)
            errores.append(abs(real_value - pred_value))
        except (OverflowError, ValueError):
            errores.append(1e5)  # Penalización alta pero manejable
    
    # Calcular el MAE
    mae = np.mean(errores)

    # Penalizar ecuaciones largas o complejas
    complexity_penalty = len(individuo) * 0.05

    return (mae + complexity_penalty,)

def stopMethod(logbook, threshold, patience):
    """
    Comprueba si el algoritmo ha convergido en función de la mejora porcentual de su mínimo
    - threshold: mejora mínima requerida para seguir buscando
    - patience: número de generaciones que deben cumplir el treshold
    """
    # Se emplea como medida de rendimiento el fitness máximo
    min_fitness = logbook.select("min")
    
    # Si hay menos de "patience" generaciones, no se puede verificar convergencia
    if len(min_fitness) < patience:
        return False

    # Se comprueba la mejora conseguida
    for i in range(1, patience):
        # Calcula la mejora porcentual
        improvement = (min_fitness[-i] - min_fitness[-i-1]) / min_fitness[-i-1]

        if abs(improvement) >= threshold:
            return False  # Continua el algoritmo

    return True # Termina el algoritmo

def plot_ind(ind, samples=50):
    """
    Representa las predicciones de un individuo en 50 estimaciones y 
    las compara con los valores reales

    Args:
        ind:     individuo a evaluar
        samples: número de muestras a evaluar
    """
    # Compilar el árbol de expresión
    func = toolbox.compile(expr=ind)
    
    # Extraer datos y resultados
    data, results = extract_data()
    data = data.values
    results = results.values.flatten()
    
    # Tomar las primeras "samples" muestras
    random_indices = np.random.choice(len(data), samples, replace=False)
    sample_data = data[random_indices]
    sample_results = results[random_indices]
    
    # Calcular predicciones
    predicciones = []
    for x in sample_data:
        try:
            pred_value = func(*x)
            predicciones.append(pred_value)
        except (OverflowError, ValueError):
            predicciones.append(None)  # Ignora la predicción
    
    # Muestra la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(range(samples), sample_results, "g-", label="Valores Reales", marker="o")
    plt.plot(range(samples), predicciones, "r--", label="Predicciones", marker="x")
    plt.title("Comparación de Valores Reales vs Predicciones")
    plt.xlabel("Muestra")
    plt.ylabel("Valor")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Árbol con nombre MAIN y aridad 14
    pset = gp.PrimitiveSet("MAIN", 14)

    # Entradas del árbol para que sean legibles en el resultado
    pset.renameArguments(ARG0='Density', ARG1='Age', ARG2='Weight', ARG3='Height',
                        ARG4='Neck', ARG5='Chest', ARG6='Abdomen', ARG7='Hip',
                        ARG8='Thigh', ARG9='Knee', ARG10='Ankle', ARG11='Biceps',
                        ARG12='Forearm', ARG13='Wrist')

    # Operaciones con dos operandos
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)  # División protegida definida arriba

    # Operaciones de un solo operando
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(protected_sqrt, 1) # Raiz protegida definida arriba

    # Constante aleatoria
    pset.addEphemeralConstant('rand101', lambda: random.randrange(-1, 1))

    # Se establece un fitness a minimizar
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # Se define la toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # Operadores genéticos
    toolbox.register("select", tools.selEpsilonLexicase, epsilon=0.5)
    toolbox.register("mate", gp.cxOnePoint)

    # Mutación y su profundidad
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Limites en altura donde se aplican los operadores genéticos
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

    # Se define cómo se evaluará cada individuo
    toolbox.register("evaluate", eval_sol)

    # Ejecutar el algoritmo genético
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Valores de población y generaciones
    population_size = 100
    num_gen = 3000
    prob_mut = 0.1
    prob_mate = 0.5
    population = toolbox.population(n=population_size)

    # Variable que recoge los resultados
    logbook = tools.Logbook()
    logbook.header = ["gen", "avg", "min", "max"]

    # Inicia el cronómetro
    start = time.time()

    # Bucle que ejecuta el algoritmo para cada genración
    for gen in range(num_gen):
        # El 5% de la población son élites
        N_elites = round(population_size*0.05)

        # Se aplica elitismo
        elites = tools.selBest(population, N_elites)

        # Se ejecuta una única generación
        population, _ = algorithms.eaSimple(
            population, toolbox,
            cxpb=prob_mate, mutpb=prob_mut,
            ngen=1, verbose=False, stats=stats
        )

        # Sustituir una parte de la población con los mejores individuos
        population[-N_elites:] = elites

        # Guardar estadísticas de la generación actual
        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        print(logbook.stream)

        # Verificar si ha habido convergencia
        if stopMethod(logbook, threshold=5e-8, patience=300):
            print(f"Convergencia alcanzada en la generación {gen}")
            break

    # Detiene el cronómetro
    end = time.time()

    # Se muestra la expresión del mejor individuo
    best = tools.selBest(population, 1)[0]
    print("\nLa mejor solución encontrada es:", best)

    # Calcula el tiempo transcurrido
    processTime = end - start
    print(f"Tiempo de procesamiento: {processTime:.6f} segundos")

    # Muestra una gráfica de predicciones frente a valores reales
    plot_ind(best)

    # Se recuperan los datos desde el log
    gen = logbook.select("gen")
    avgs = logbook.select("avg")
    maxim = logbook.select("max")
    minim = logbook.select("min")

    # Escalar los datos del logbook
    max_fitness = max(maxim)
    min_fitness = min(minim)

    # Se establecen dos figuras para graficar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico 1: Mínimos
    ax1.plot(gen, minim, "b-", label="Min Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.set_title("Fitness mínimo")
    ax1.grid(True)
    ax1.legend()

    # Gráfico 2: Promedios y Máximos
    ax2.plot(gen, avgs, "g-", label="Average Fitness")
    ax2.plot(gen, maxim, "r-", label="Max Fitness")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness")
    ax2.set_title("Fitness promedio y máximo")
    ax2.grid(True)
    ax2.legend()

    # Mostrar ambos gráficos
    plt.tight_layout()
    plt.show()