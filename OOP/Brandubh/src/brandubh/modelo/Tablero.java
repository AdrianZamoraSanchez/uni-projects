package brandubh.modelo;
import java.util.Arrays;

import brandubh.util.*;

/**
 * Clase que implementa un tablero con una celda en cada posible
 * posicion, además de métodos para el control e interacción con el tablero
 * 
 * @author Adrián Zamora Sánchez / azs1004@alu.ubu.es
 * @see brandubh.control.Arbitro
 * @version 1.0
 * @since 1.0
*/
public class Tablero {
	/**
	 * Matriz de las celdas del tablero, será de dos dimensiones 
	 * de longitud variable
	 * 
	 * @see brandubh.modelo.Celda
	 */
	private Celda matriz[][];
	
	/**
	 * Constructor del tablero, genera el tablero con 7x7 celdas,
	 * se encarga de establecer las celdas especiales como el trono
	 * y las provincias en el tablero por defecto
	 */
	public Tablero() {
		// Variable que contiene el número de filas y columnas del tablero cuadrado
		final int dimensionesTablero = 7; // El tablero es de dimensionesTablero x dimensionesTablero
		
		// Define la matriz
		matriz = new Celda[dimensionesTablero][dimensionesTablero];
		
		// Inicialización de las celdas normales del tablero
        for (int i = 0; i < dimensionesTablero; i++) {
            for (int j = 0; j < dimensionesTablero; j++) {
            	if(matriz[i][j] == null) {
                    matriz[i][j] = new Celda(new Coordenada(i,j));
            	}
            }
        }
        
	    // Inicialización de las celdas especiales del tablero
		// Trono
		matriz[3][3] = new Celda(new Coordenada(3,3), TipoCelda.TRONO);
		
		// Provincias
		matriz[0][0] = new Celda(new Coordenada(0,0), TipoCelda.PROVINCIA);
		matriz[6][0] = new Celda(new Coordenada(6,0), TipoCelda.PROVINCIA);
		matriz[0][6] = new Celda(new Coordenada(0,6), TipoCelda.PROVINCIA);
		matriz[6][6] = new Celda(new Coordenada(6,6), TipoCelda.PROVINCIA);
	}
	
	/**
	 * Devuelve un string con los datos del tablero
	 * 
	 * @return String
	 */
	public String aTexto() {
	    StringBuilder string = new StringBuilder();
	    int numFilas = consultarNumeroFilas();
	    int numCols = consultarNumeroColumnas();

	    // Contenido del tablero
	    for (int i = 0; i < numFilas; i++) {
	        // Números laterales
	    	string.append(numFilas-i);
	        
	    	// Celdas del tablero
	        for (int j = 0; j < numCols; j++) {
	            Pieza pieza = obtenerCelda(new Coordenada(i, j)).consultarPieza();
	            if(pieza != null) {
	            	 string.append(pieza.consultarTipoPieza().toChar());
	            }else {
	            	string.append("-");
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
		for(int i = 0; i < matriz.length; i++) {
			for(int j = 0; j < matriz[0].length; j++) {
				// Comprueba si cada celda está vacía
				if(matriz[i][j].estaVacia()) {
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
		// Variable que cuenta el numero de piezas
		int contador = 0;
		
		// Recorre todas las celdas y compara la pieza que contienen con el tipo buscado
		for(int i = 0; i < matriz.length; i++) {
			for(int j = 0; j < matriz[0].length; j++) {
				if(!matriz[i][j].estaVacia() && matriz[i][j].consultarPieza().consultarTipoPieza() == tipo) {
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
	private boolean comprobarCoordenadas(Coordenada coordenadas) {
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
	 * @param coordenadas ubicación de la pieza a eliminar
	 */
	public void eliminarPieza(Coordenada coordenadas) {
		// Comprueba que sean coordenadas correctas
		if(!comprobarCoordenadas(coordenadas)) {
			return;
		}
		
		// Elimina la pieza
		matriz[coordenadas.fila()][coordenadas.columna()].eliminarPieza();
	}
	
	/**
	 * Devuelve un clon de una celda específica dadas unas coordenadas
	 * 
	 * @param coordenadas coordenadas a consultar
	 * @return Celda
	 */
	public Celda consultarCelda(Coordenada coordenadas) {
		// Comprueba que sean coordenadas correctas
		if(!comprobarCoordenadas(coordenadas)) {
			return null;
		}
		
		// Devuelve la celda consultada
		return matriz[coordenadas.fila()][coordenadas.columna()].clonar();
	}
	
	/**
	 * Devuelve un array con un clon de todas las celdas del tablero
	 * 
	 * @return Celda[]
	 */
	public Celda[] consultarCeldas() {
		// Variable con las dimensiones del tablero, al ser cuadrado solo hace falta una dimension
		int dimensionesTablero = matriz.length;
		
		// Se define un array del tamaño del tablero completo
		Celda[] arrayCeldas = new Celda[dimensionesTablero*dimensionesTablero];
		
		// Se recorren ambas dimensiones del tablero añadiendo linealmente en el array las celdas clonadas
		for(int i = 0; i < dimensionesTablero; i++) {
			for(int j = 0; j < dimensionesTablero; j++) {
					arrayCeldas[i*dimensionesTablero + j] = matriz[i][j].clonar();
			}
		}
		
		// Devuelve el array con las celdas clonadas
		return arrayCeldas;
	}
	
	/**
	 * Devuelve un array con un clon de todas las celdas contiguas 
	 * a la coordenada especificada en el tablero
	 * 
	 * @param coordenada coordenada que ubica la consulta
	 * @return Celda[]
	 */
	public Celda[] consultarCeldasContiguas(Coordenada coordenada) {
		// Comrprueba que sea una coordenada valida y no nula
		if(!comprobarCoordenadas(coordenada)){
			// En caso de coordenada incorrecta devuelve un array vacio
			Celda[] arrayCeldas = new Celda[0];
			return arrayCeldas;
		}
		
		// Se toman las celdas en horizontal y vertical
		Celda[] arrayCeldasHorizontal = consultarCeldasContiguasEnHorizontal(coordenada);
		Celda[] arrayCeldasVertical = consultarCeldasContiguasEnVertical(coordenada);
		
		// Se define el array que contendrá las celdas tanto en vertical como en horizontal
		int numeroCeldas = arrayCeldasHorizontal.length + arrayCeldasVertical.length;
		Celda[] arrayCeldas = new Celda[numeroCeldas];
		
		// Se utiliza un contador para combinar ambos arrays
		int contador = 0;
		
		// Se pasan las celdas del array de celdas en horizontal al array en ambas direcciones
		for(Celda celda : arrayCeldasHorizontal) {
			arrayCeldas[contador] = celda;
			contador++;
		}
		
		// Se pasan las celdas del array de celdas en vertical al array en ambas direcciones
		for(Celda celda : arrayCeldasVertical) {
			arrayCeldas[contador] = celda;
			contador++;
		}
		
		// Se devulve el array con todas las celdas
		return arrayCeldas;
	}
	
	/**
	 * Devuelve un array con un clon de todas las celdas adyacentes horizontalmente
	 * a la coordenada especificada en el tablero
	 * 
	 * @param coordenada coordenada que ubica la consulta
	 * @return Celda[]
	 */
	public Celda[] consultarCeldasContiguasEnHorizontal(Coordenada coordenada) {
		// Array que contendrá las celdas
		Celda[] arrayCeldas;
		
		// Si la coordenada no es valida se devuelve un array vacío
		if(!comprobarCoordenadas(coordenada)) {
			arrayCeldas = new Celda[0];
			return arrayCeldas;
		}

		// Comprueba si la coordenada se encuentra en el borde lateral del tablero (columnas 0  o 6)
		if(coordenada.columna() <= 0 && coordenada.columna() < 6) {
			// Se define el array con tamaño 1
			arrayCeldas = new Celda[1];
			
			// En caso de estar en el extremo izquierdo, solo devuelve la celda de la derecha
			arrayCeldas[0] = matriz[coordenada.fila()][coordenada.columna()+1].clonar();
		}else if(coordenada.columna() >= 6 && coordenada.columna() > 1) {
			// Se define el array con tamaño 1
			arrayCeldas = new Celda[1];
			
			// En caso de estar en el extremo derecho, solo devuelve la celda de la izquierda
			arrayCeldas[0] = matriz[coordenada.fila()][coordenada.columna()-1].clonar();
		}else {
			// Se define el array con tamaño 2
			arrayCeldas = new Celda[2];
			
			// En caso de encontrarse lejos de los extremos del tablero devuelve celda por la izquierda y derecha
			arrayCeldas[0] = matriz[coordenada.fila()][coordenada.columna()+1].clonar();
			arrayCeldas[1] = matriz[coordenada.fila()][coordenada.columna()-1].clonar();
		}
		
		return arrayCeldas;
	}
	
	/**
	 * Devuelve un array con un clon de todas las celdas adyacentes verticalmente
	 * a la coordenada especificada en el tablero
	 * 
	 * @param coordenada coordenada que ubica la consulta
	 * @return Celda[]
	 */
	public Celda[] consultarCeldasContiguasEnVertical(Coordenada coordenada) {
		// Array que contendrá las celdas
		Celda[] arrayCeldas;
		
		// Si la coordenada no es valida se devuelve un array vacío
		if(!comprobarCoordenadas(coordenada)) {
			arrayCeldas = new Celda[0];
			return arrayCeldas;
		}
		
		// Comprueba si la coordenada se encuentra en el borde superior del tablero (filas 0  o 6)
		if(coordenada.fila() <= 0 && coordenada.fila() < 6) {
			// Se define el array con tamaño 1
			arrayCeldas = new Celda[1];
						
			// En caso de estar en el extremo superior, solo devuelve la celda de abajo
			arrayCeldas[0] = matriz[coordenada.fila()+1][coordenada.columna()].clonar();
		}else if(coordenada.fila() >= 6 && coordenada.fila() > 1) {
			// Se define el array con tamaño 1
			arrayCeldas = new Celda[1];
						
			// En caso de estar en el extremo inferior, solo devuelve la celda de arriba
			arrayCeldas[0] = matriz[coordenada.fila()-1][coordenada.columna()].clonar();
		}else {
			// Se define el array con tamaño 2
			arrayCeldas = new Celda[2];
			
			// En caso de encontrarse lejos de los extremos del tablero devuelve celda de arriba y abajo
			arrayCeldas[0] = matriz[coordenada.fila()+1][coordenada.columna()].clonar();
			arrayCeldas[1] = matriz[coordenada.fila()-1][coordenada.columna()].clonar();
		}
		
		return arrayCeldas;
	}
	
	/**
	 * Devuelve el número de columnas del tablero
	 * 
	 * @return int
	 */
	public int consultarNumeroColumnas() {
		return matriz[0].length;
	}
	
	/**
	 * Devuelve el número de filas del tablero
	 * 
	 * @return int
	 */
	public int consultarNumeroFilas() {
		return matriz.length;
	}
	
	/**
	 * Coloca una pieza en el tablero
	 * 
	 * @param pieza pieza a colorcar
	 * @param coordenada coordenada de la celda donde se coloca la pieza
	 */
	public void colocar(Pieza pieza, Coordenada coordenada){
		// Comprueba la validez de las coordenadas y la existencia de la pieza
		if(pieza == null || !comprobarCoordenadas(coordenada)) {
			return;
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
	 */
	public boolean estaEnTablero(Coordenada coordenada) {
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
	 */
	public Celda obtenerCelda(Coordenada coordenada) {
		// Si la coordenada no es valida devuelve null
		if(!comprobarCoordenadas(coordenada)){
			return null;
		}
		
		// Devuelve la referencia a la celda
		return matriz[coordenada.fila()][coordenada.columna()];
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
        Celda[][] nuevaMatriz = new Celda[matriz.length][matriz[0].length];
        
        // Si el objeto que se clona tiene una matriz la clona
        if(matriz != null) {
        	// Recorre el tablero y clona cada celda
        	for (int i = 0; i < 7; i++) {
	            for (int j = 0; j < 7; j++) {
	                nuevaMatriz[i][j] = matriz[i][j].clonar();
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
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.deepHashCode(matriz);
		return result;
	}

	/**
	 * Devuelve true si es un tablero igual y false si no lo es
	 * 
	 * @param obj objeto con el que se hace la comparación
	 * @return boolean
	 */
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Tablero other = (Tablero) obj;
		return Arrays.deepEquals(matriz, other.matriz);
	}
}