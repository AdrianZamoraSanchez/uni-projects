package tafl.util;

/**
 * Record que almacena dos int, para fila y columna representando
 * las posibles coordenadas en el tablero
 * 
 * @author <a href="azs1004@alu.ubu.es">Adrián Zamora Sánchez</a>
 * @see tafl.modelo.Celda
 * @see tafl.modelo.Tablero
 * @version 1.0
 * @since 1.0
 * @param fila corresponde al eje x
 * @param columna corresponde al eje y
 * 
*/
public record Coordenada(int fila, int columna) {
	
	/**
	 * Función que devuelve las características de la coordenada en formato texto
	 * 
	 * @return String
	 */
	public String toString() {
		return "Coordenada[fila=" + this.fila + ", columna=" + this.columna + "]";
	}
}