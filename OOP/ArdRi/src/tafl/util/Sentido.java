package tafl.util;

/**
 * Enum que lista las posibles direcciones en las que se realiza
 * una jugada
 * 
 * @author <a href="azs1004@alu.ubu.es">Adrián Zamora Sánchez</a>
 * @see tafl.modelo.Pieza
 * @version 1.0
 * @since 1.0
*/
public enum Sentido {
	/** Vertical en el sentido Norte */
	VERTICAL_N(-1, 0),
	/** Vertical en sentido Sur */
	VERTICAL_S(1, 0),
	/** Horizontal en el sentido Oeste*/
	HORIZONTAL_O(0,-1),
	/** Horizontal en el sentido Este */
	HORIZONTAL_E(0,1);
	
	/**
	 * Almacena el desplazamiento en el eje X
	 * 
	 * @see java.lang.Object
	 */
	private int desplazamientoEnFilas;
	
	/**
	 * Almacena el desplazamiento en el eje Y
	 * 
	 * @see java.lang.Object
	 */
	private int desplazamientoEnColumnas;
	
	/**
	 * Constructor de la clase Sentido, toma los enteros que forman las 
	 * coordenadas del desplazamiento y las asignan a los atributos correspondientes
	 * 
	 * @param valorDesplazamientoFilas 		numero de filas que se desplaza
	 * @param valorDesplazamientoColumnas 	numero de columnas que se desplaza
	 */
	private Sentido(int valorDesplazamientoFilas, int valorDesplazamientoColumnas) {
		desplazamientoEnFilas = valorDesplazamientoFilas;
		desplazamientoEnColumnas = valorDesplazamientoColumnas;
	}
	
	/**
	 * Devuelve el número de filas desplazadas
	 * 
	 * @return int
	 */
	public int consultarDesplazamientoEnFilas() {
		return this.desplazamientoEnFilas;
	}
	
	/**
	 * Devuelve el número de columnas desplazadas
	 * 
	 * @return int
	 */
	public int consultarDesplazamientoEnColumnas() {
		return this.desplazamientoEnColumnas;
	}
}
