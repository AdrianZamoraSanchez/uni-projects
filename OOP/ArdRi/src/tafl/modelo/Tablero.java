package tafl.modelo;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import tafl.excepcion.CoordenadasIncorrectasException;
import tafl.util.*;

/**
 * Clase que implementa un tablero con una celda en cada posible
 * posicion, además de métodos para el control e interacción con el tablero
 * 
 * @author <a href="azs1004@alu.ubu.es">Adrián Zamora Sánchez</a>
 * @see tafl.control.Arbitro
 * @version 1.0
 * @since 1.0
*/
public class Tablero {
	/**
	 * Matriz de las celdas del tablero, será de dos dimensiones 
	 * de longitud variable
	 * 
	 * @see tafl.modelo.Celda
	 */
	private List<List<Celda>> matriz;
	
	/**
	 * Constructor del tablero, genera el tablero con 7x7 celdas,
	 * se encarga de establecer las celdas especiales como el trono
	 * y las provincias en el tablero por defecto
	 */
	public Tablero() {
		// Variable que contiene el número de filas y columnas del tablero cuadrado
		final int dimensionesTablero = 7; // El tablero es de dimensionesTablero x dimensionesTablero
		
		// Define la lista de listas
		matriz = new ArrayList<>();
		
		// Genera las filas
		for (int i = 0; i < dimensionesTablero; i++) {
			// Agrega una nueva lista a cada fila
			matriz.add(new ArrayList<>());
		}
		
		// Inicialización de las celdas normales del tablero
        for (int i = 0; i < dimensionesTablero; i++) {
            for (int j = 0; j < dimensionesTablero; j++) {
            	// Añade una celda normal
            	matriz.get(i).add(new Celda(new Coordenada(i, j)));
            }
        }
        
	    // Inicialización de las celdas especiales del tablero
		// Trono
        matriz.get(3).set(3, new Celda(new Coordenada(3,3), TipoCelda.TRONO));
		
		// Provincias
        matriz.get(0).set(0, new Celda(new Coordenada(0,0), TipoCelda.PROVINCIA));
        matriz.get(6).set(0, new Celda(new Coordenada(6,0), TipoCelda.PROVINCIA));
		matriz.get(0).set(6, new Celda(new Coordenada(0,6), TipoCelda.PROVINCIA));
		matriz.get(6).set(6, new Celda(new Coordenada(6,6), TipoCelda.PROVINCIA));
	}
	
	/**
	 * Devuelve un string con los datos del tablero
	 * 
	 * @return String
	 */
	public String aTexto(){
	    StringBuilder string = new StringBuilder();
	    int numFilas = consultarNumeroFilas();
	    int numCols = consultarNumeroColumnas();

	    // Contenido del tablero
	    for (int i = 0; i < numFilas; i++) {
	        // Números laterales
	    	string.append(numFilas-i);
	        
	    	// Celdas del tablero
	        for (int j = 0; j < numCols; j++) {
	        	try {
	        		// Obtiene las piezas del tablero
	                Pieza pieza = obtenerCelda(new Coordenada(i, j)).consultarPieza();
	                
	                // Si la pieza no es nula coloca su caracter correspondiente, en caso de nula coloca "-"
	                if (pieza != null) {
	                    string.append(pieza.consultarTipoPieza().toChar());
	                } else {
	                    string.append("-");
	                }
	            } catch (CoordenadasIncorrectasException e) {
	                // En caso de error al obtener una pieza continua con la siguiente iteración del bucle
	                continue;
	            }
	        }
	        string.append("\n");
	    }
	    
	    // Espacio en blanco para cuadrar las columnas y las letras
	    string.append(' '); 
	    
	    // Índices de las columnas
	    for (int j = 0; j < numCols; j++) {
	        string.append((char) ('a' + j));
	    }
	    string.append("\n");

	    // Se devuelve el string que identifica el tablero
	    return string.toString();
	}
	
	/**
	 * Devuelve true si el tablero tiene todas sus celdas rellenas
	 * con piezas y false si hay hueco para colocar más piezas
	 * 
	 * @return boolean
	 */
	public boolean estaCompleto(){
		// Doble bucle que recorre el tablero entero
		for(int i = 0; i < matriz.size(); i++) {
			for(int j = 0; j < matriz.get(0).size(); j++) {
				// Comprueba si cada celda está vacía
				if(matriz.get(i).get(j).estaVacia()) {
					return false;
				}
			}
		}
		return true;
	}
	
	/**
	 * Devuelve el número de piezas en el tablero con un color concreto
	 * 
	 * @param tipo tipo de piezas a contar
	 * @return int
	 */
	public int consultarNumeroPiezas(TipoPieza tipo) {
		// Si el parámetro es nulo lanza una excepción por parametro incorrecto
		if(tipo == null) {
			throw new IllegalArgumentException("Parámetro TipoPieza nulo");
		}
				
		// Variable que cuenta el numero de piezas
		int contador = 0;
		
		// Recorre todas las celdas y compara la pieza que contienen con el tipo buscado
		for(int i = 0; i < matriz.size(); i++) {
			for(int j = 0; j < matriz.get(0).size(); j++) {
				if(!matriz.get(i).get(j).estaVacia() && matriz.get(i).get(j).consultarPieza().consultarTipoPieza() == tipo) {
					// Si encuentra una pieza de igual tipo aumenta el contador
					contador++;
				}
			}
		}
		
		// Devuelve el valor del contador
		return contador;
	}
	
	/**
	 * Comprueba si son coordenadas validas en el tablero
	 * 
	 * @param coordenadas coordenadas a comprobar
	 * @return boolean
	 */
	public boolean comprobarCoordenadas(Coordenada coordenadas) {
		// Comprueba que se pase una coordenada
		if(coordenadas == null) {
			return false;
		}
		
		// Comprueba la validez de la fila, debe estar dentro del tablero
		if(coordenadas.fila() < 0 || coordenadas.fila() >= consultarNumeroFilas()) {
			return false;
		}
		
		// Comprueba la validez de la fila, debe estar dentro del tablero
		if(coordenadas.columna() < 0 || coordenadas.columna() >= consultarNumeroColumnas()) {
			return false;
		}
		
		// Devuelve true, pues ambas coordenadas son correctas
		return true;
	}
	
	/**
	 * Elimina la pieza en la coordenada especificada
	 * 
	 * @param coordenada ubicación de la pieza a eliminar
	 * @throws CoordenadasIncorrectasException excepción de coordenadas incorrectas
	 */
	public void eliminarPieza(Coordenada coordenada) throws CoordenadasIncorrectasException {
		// Comprueba la exitencia de los parametros necesarios
		if(coordenada == null) {
			throw new IllegalArgumentException("Parámetro coordenada nulo en la eliminación de una pieza");
		}
		
		// Si las coordenadas no son correctas lanza una excepción indicandolo
		if(!comprobarCoordenadas(coordenada)) {
			throw new CoordenadasIncorrectasException("Coordenadas proporcionadas incorrectas para eliminar una pieza");
		}
				
		// Comprueba que sean coordenadas correctas
		if(!comprobarCoordenadas(coordenada)) {
			return;
		}
		
		// Elimina la pieza
		matriz.get(coordenada.fila()).get(coordenada.columna()).eliminarPieza();;
	}
	
	/**
	 * Devuelve un clon de una celda específica dadas unas coordenadas
	 * 
	 * @param coordenada coordenadas a consultar
	 * @return Celda
	 * @throws CoordenadasIncorrectasException excepción de coordenadas incorrectas
	 */
	public Celda consultarCelda(Coordenada coordenada) throws CoordenadasIncorrectasException {
		// Si la coordenada es nula lanza una excepción de parametro ilegal
		if(coordenada == null) {
			throw new IllegalArgumentException("Falta parámetro coordenada para consultar una celda");
		}
		
		// Si la coordenada no es lanza una excepción de coordenadas incorrectas
		if(!comprobarCoordenadas(coordenada)) {
			throw new CoordenadasIncorrectasException("Coordenadas incorrectas en la consulta de una celda del tablero");
		}
				
		// Comprueba que sean coordenadas correctas
		if(!comprobarCoordenadas(coordenada)) {
			return null;
		}
		
		// Devuelve la celda consultada
		return matriz.get(coordenada.fila()).get(coordenada.columna()).clonar();
	}
	
	/**
	 * Devuelve un array con un clon de todas las celdas del tablero
	 * 
	 * @return List
	 */
	public List<Celda> consultarCeldas() {
		// Se define un array del tamaño del tablero completo
		List<Celda> lista = new ArrayList<Celda>();
		
		// Se recorren ambas dimensiones del tablero añadiendo linealmente en el array las celdas clonadas
		for(int i = 0; i < matriz.size(); i++) {
			for(int j = 0; j < matriz.get(0).size(); j++) {
				lista.add(matriz.get(i).get(j).clonar());
			}
		}
		
		// Devuelve el array con las celdas clonadas
		return lista;
	}
	
	/**
	 * Devuelve un array con un clon de todas las celdas contiguas 
	 * a la coordenada especificada en el tablero
	 * 
	 * @param coordenada coordenada que ubica la consulta
	 * @return List
	 * @throws CoordenadasIncorrectasException excepción de coordenadas incorrectas
	 */
	public List<Celda> consultarCeldasContiguas(Coordenada coordenada) throws CoordenadasIncorrectasException {
		// Lista que contendrá las celdas
	    List<Celda> listaCeldas = new ArrayList<>();

	    // Se toman las celdas en horizontal y vertical
	    List<Celda> listaCeldasHorizontal = consultarCeldasContiguasEnHorizontal(coordenada);
	    List<Celda> listaCeldasVertical = consultarCeldasContiguasEnVertical(coordenada);

	    // Se combinan las listas en una sola
	    listaCeldas.addAll(listaCeldasHorizontal);
	    listaCeldas.addAll(listaCeldasVertical);

	    return listaCeldas;
	}
	
	/**
	 * Devuelve un array con un clon de todas las celdas adyacentes horizontalmente
	 * a la coordenada especificada en el tablero
	 * 
	 * @param coordenada coordenada que ubica la consulta
	 * @return List
	 * @throws CoordenadasIncorrectasException excepción de coordenadas incorrectas
	 */
	public List<Celda> consultarCeldasContiguasEnHorizontal(Coordenada coordenada) throws CoordenadasIncorrectasException {
		// Si la coordenada es nula lanza una excepción de parametro ilegal
		if(coordenada == null) {
			throw new IllegalArgumentException("Falta parámetro coordenada al consultar celdas en horizontal");
		}
		
		// Si la coordenada no es lanza una excepción de coordenadas incorrectas
		if(!comprobarCoordenadas(coordenada)) {
			throw new CoordenadasIncorrectasException("Coordenadas incorrectas al consultar celdas en horizontal");
		}
		
		// Lista que contendrá las celdas
	    List<Celda> listaCeldas = new ArrayList<>();
	
	    // Comprueba si la coordenada se encuentra en el borde lateral del tablero (columnas 0 o 6)
	    if (coordenada.columna() <= 0 && coordenada.columna() < 6) {
	        // En caso de estar en el extremo izquierdo, solo agrega la celda de la derecha
	        listaCeldas.add(matriz.get(coordenada.fila()).get(coordenada.columna() + 1).clonar());
	    } else if (coordenada.columna() >= 6 && coordenada.columna() > 1) {
	        // En caso de estar en el extremo derecho, solo agrega la celda de la izquierda
	        listaCeldas.add(matriz.get(coordenada.fila()).get(coordenada.columna() - 1).clonar());
	    } else {
	        // En caso de encontrarse lejos de los extremos del tablero agrega celda por la izquierda y derecha
	        listaCeldas.add(matriz.get(coordenada.fila()).get(coordenada.columna() + 1).clonar());
	        listaCeldas.add(matriz.get(coordenada.fila()).get(coordenada.columna() - 1).clonar());
	    }
	
	    return listaCeldas;
	}

	
	/**
	 * Devuelve un array con un clon de todas las celdas adyacentes verticalmente
	 * a la coordenada especificada en el tablero
	 * 
	 * @param coordenada coordenada que ubica la consulta
	 * @return List
	 * @throws CoordenadasIncorrectasException excepción de coordenadas incorrectas
	 */
	public List<Celda> consultarCeldasContiguasEnVertical(Coordenada coordenada) throws CoordenadasIncorrectasException {
		// Si la coordenada es nula lanza una excepción de parametro ilegal
		if(coordenada == null) {
			throw new IllegalArgumentException("Falta parámetro coordenada al consultar celdas en vertical");
		}
		
		// Si la coordenada no es lanza una excepción de coordenadas incorrectas
		if(!comprobarCoordenadas(coordenada)) {
			throw new CoordenadasIncorrectasException("Coordenadas incorrectas al consultar celdas en vertical");
		}
		
		// Lista que contendrá las celdas
	    List<Celda> listaCeldas = new ArrayList<>();

	    // Comprueba si la coordenada se encuentra en el borde superior del tablero (filas 0 o 6)
	    if (coordenada.fila() <= 0 && coordenada.fila() < 6) {
	        // En caso de estar en el extremo superior, solo agrega la celda de abajo
	        listaCeldas.add(matriz.get(coordenada.fila() + 1).get(coordenada.columna()).clonar());
	    } else if (coordenada.fila() >= 6 && coordenada.fila() > 1) {
	        // En caso de estar en el extremo inferior, solo agrega la celda de arriba
	        listaCeldas.add(matriz.get(coordenada.fila() - 1).get(coordenada.columna()).clonar());
	    } else {
	        // En caso de encontrarse lejos de los extremos del tablero agrega celda de arriba y abajo
	        listaCeldas.add(matriz.get(coordenada.fila() + 1).get(coordenada.columna()).clonar());
	        listaCeldas.add(matriz.get(coordenada.fila() - 1).get(coordenada.columna()).clonar());
	    }

	    return listaCeldas;
	}
	
	/**
	 * Devuelve el número de columnas del tablero
	 * 
	 * @return int
	 */
	public int consultarNumeroColumnas() {
		return matriz.get(0).size();
	}
	
	/**
	 * Devuelve el número de filas del tablero
	 * 
	 * @return int
	 */
	public int consultarNumeroFilas() {
		return matriz.size();
	}
	
	/**
	 * Coloca una pieza en el tablero
	 * 
	 * @param pieza pieza a colorcar
	 * @param coordenada coordenada de la celda donde se coloca la pieza
	 * @throws CoordenadasIncorrectasException excepción de coordenadas incorrectas
	 */
	public void colocar(Pieza pieza, Coordenada coordenada) throws CoordenadasIncorrectasException{
		// Comprueba la exitencia de los parametros necesarios
		if(pieza == null || coordenada == null) {
			throw new IllegalArgumentException("Parámetros nulos al colocar una pieza en el tablero");
		}
		
		// Si las coordenadas no son correctas lanza una excepción indicandolo
		if(!comprobarCoordenadas(coordenada)) {
			throw new CoordenadasIncorrectasException("Coordenadas incorrectas al colocar una pieza en el tablero");
		}
		
		// Toma la referencia de la celda y coloca la pieza
		obtenerCelda(coordenada).colocar(pieza);
	}
	
	/**
	 * Devuelve true si las coordenadas especificadas se encuentran en el tablero
	 * y false si son una coordenadas que no se encuentran en el tablero
	 * 
	 * @param coordenada coordenada que se quiere comprobar
	 * @return boolean
	 * @throws CoordenadasIncorrectasException excepción por coordenadas incorrectas
	 */
	public boolean estaEnTablero(Coordenada coordenada) throws CoordenadasIncorrectasException {
		if(coordenada == null) {
			throw new IllegalArgumentException();
		}
		
		// Si la coordenada no es correcta devuelve false (no puede haber una pieza fuera del tablero)
		if(!comprobarCoordenadas(coordenada)) {
			return false;
		}
		
		// Coprueba si la celda se encuentra en el tablero
		if(obtenerCelda(coordenada) != null) {
			return true;
		}
		
		// Encaso de no poder obtener la celda devuelve null
		return false;
	}
	
	/**
	 * Devuelve la celda de una coordenadas dadas
	 * 
	 * @param coordenada coordenada de la celda que se quire obtener
	 * @return Celda celda perteneciente a las coordenadas indicadas
	 * @throws CoordenadasIncorrectasException excepción por coordenadas incorrectas
	 */
	public Celda obtenerCelda(Coordenada coordenada) throws CoordenadasIncorrectasException {
		// Comprueba la exitencia del parametro necesario
		if(coordenada == null) {
			throw new IllegalArgumentException("Falta parámetrocoordenada necesario para obtener una celda");
		}
		
		// Si las coordenadas no son correctas lanza una excepción indicandolo
		if(!comprobarCoordenadas(coordenada)) {
			throw new CoordenadasIncorrectasException("Coordenadas proporcionadas incorrectas para obtener una celda del tablero");
		}
		
		// Devuelve la referencia a la celda
		return matriz.get(coordenada.fila()).get(coordenada.columna());
	}
	
	/**
	 * Clona el tablero y sus atributos y devuelve un tablero
	 * exactamente igual
	 * 
	 * @return Tablero
	 */
	public Tablero clonar() {
	    // Crea un nuevo tablero
	    Tablero clon = new Tablero();
	    
	    // Crea la matriz que será atributo del clon
	    List<List<Celda>> nuevaMatriz = new ArrayList<>();
	    
	    // Genera las filas
 		for (int i = 0; i < matriz.size(); i++) {
 			// Agrega una nueva lista a cada fila
 			nuevaMatriz.add(new ArrayList<>());
 		}
	    
	    // Si el objeto que se clona tiene una matriz, la clona
	    if (matriz != null) {
	        // Recorre el tablero y clona cada celda
	        for (int i = 0; i < matriz.size(); i++) {
	            for (int j = 0; j < matriz.get(0).size(); j++) {
	            	// Si no es nula añade la celda, en caso de serlo la deja nula
	                if (matriz.get(i).get(j) != null) {
	                    nuevaMatriz.get(i).add(matriz.get(i).get(j).clonar());
	                } else {
	                    nuevaMatriz.get(i).add(null);
	                }
	            }
	        }
	        
	        // Asigna la matriz de celdas al clon
	        clon.matriz = nuevaMatriz;
	    }
	    
	    // Devuelve un clon del tablero
	    return clon;
	}

	/**
	 * Devuelve el codigo hash que identifica unequivocamente cada instancia de este objeto
	 * 
	 * @return int codigo hash
	 */
	@Override
	public int hashCode() {
		// Devuelve el hash de la matriz
		return Objects.hash(matriz);
	}

	/**
	 * Devuelve true si es un tablero igual y false si no lo es
	 * 
	 * @param obj objeto con el que se hace la comparación
	 * @return boolean
	 */
	@Override
	public boolean equals(Object obj) {
		// Se hace las comparaciones previas
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		
		Tablero other = (Tablero) obj;
		
		// Devuelve el resultado de la comparación
		return Objects.equals(matriz, other.matriz);
	}
}