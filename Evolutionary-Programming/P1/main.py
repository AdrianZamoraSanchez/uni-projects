# %% [markdown]
# * Endpoints:
# 
# Cada endpoint es una lista con dos elementos:  
#     - datacenter_latency: La latencia desde el centro de datos hasta el endpoint.  
#     - cache_connections: Una sublista que contiene pares de [cache_id, cache_latency] para cada caché conectado al endpoint.  
# 
# * Requests:
# 
# Cada solicitud es una lista con tres elementos:  
#     - video_id: ID del video solicitado.  
#     - endpoint_id: ID del endpoint desde el que se solicita el video.  
#     - num_requests: Número de solicitudes para ese video desde ese endpoint.  
# 
# * Parametro de Retorno:
# 
# La función devuelve un diccionario que contiene todos los datos relevantes, con endpoints y requests como matrices. Cada fila en endpoints y requests es una lista con los atributos asociados
# 

# %%
def read_input(filename):
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

        # Leer la información de cada endpoint y almacenarla en una matriz
        endpoints = []
        for _ in range(num_endpoints):
            # Leer latencia al centro de datos y cantidad de cachés conectados
            endpoint_info = file.readline().strip().split()
            datacenter_latency = int(endpoint_info[0])
            num_connected_caches = int(endpoint_info[1])
            
            # Crear una lista para almacenar las conexiones de cachés como pares [ID de caché, latencia]
            cache_connections = []
            for _ in range(num_connected_caches):
                cache_id, cache_latency = map(int, file.readline().strip().split())
                cache_connections.append([cache_id, cache_latency])
            
            # Almacenar el endpoint en formato de lista: [latencia al centro de datos, conexiones a cachés]
            endpoints.append([datacenter_latency, cache_connections])

        # Leer las solicitudes de video y almacenarlas en una matriz
        requests = []
        for _ in range(num_requests):
            video_id, endpoint_id, num_requests = map(int, file.readline().strip().split())
            # Cada solicitud es una lista con [ID del video, ID del endpoint, número de solicitudes]
            requests.append([video_id, endpoint_id, num_requests])

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

# Ejemplo de uso

def main():

    filename = "..\\qualification_round_2017.in\\streaming\\example.txt"  # Cambia esto al nombre de tu archivo de entrada
    data = read_input(filename)

    # Imprimir datos para verificar
    print("Número de videos:", data["num_videos"])
    print("Número de endpoints:", data["num_endpoints"])
    print("Número de solicitudes:", data["num_requests"])
    print("Número de cachés:", data["num_caches"])
    print("Capacidad de caché:", data["cache_capacity"])
    print("Tamaños de videos:", data["video_sizes"])
    print("Endpoints:", data["endpoints"])
    print("Solicitudes:", data["requests"])

if __name__ == "__main__":
    main()