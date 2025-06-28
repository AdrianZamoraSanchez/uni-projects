package brandubh.modelo;

import brandubh.util.*;

/**
 * Clase que implementa las jugadas, las cuales se componen del movimiento
 * de una pieza de una celda de origen a una celda de destino
 * 
 * @author <a href="azs1004@alu.ubu.es">Adrián Zamora Sánchez</a>
 * @see brandubh.control.Arbitro
 * @version 1.0
 * @since 1.0
 * 
*/
public class Jugada{
	/**
	 * Celda origen de la jugada
	 * 
	 * @see brandubh.modelo.Celda
	 */
	private Celda origen;
	
	/**
	 * Celda destino de la jugada
	 * 
	 * @see brandubh.modelo.Celda
	 */
	private Celda destino;
	
	/**
	 * Constructor de la jugada, requiere de una celda de origen y otra de destino
	 * 
	 * @param origenJugada celda de origen
	 * @param destinoJugada celda de destino
	 */
	public Jugada(Celda origenJugada, Celda destinoJugada) {
		origen = origenJugada;
		destino = destinoJugada;
	}
	
	/**
	 * Devuelve la celda de origen de la jugada
	 * 
	 * @return Celda
	 */
	public Celda consultarOrigen() {
		return origen;
	}
	
	/**
	 * Devuelve la celda de destino de la jugada
	 * 
	 * @return Celda
	 */
	public Celda consultarDestino() {
		return destino;
	}
	
	/**
	 * Devuelve true si es un movimiento ortogonal (en un solo eje)
	 * 
	 * @return boolean
	 */
	public boolean esMovimientoHorizontalOVertical() {
		// Toma las referencias a las coordenadas de origen y destino
		int origenFila = origen.consultarCoordenada().fila();
		int origenColumna = origen.consultarCoordenada().columna();
		
		int destinoFila = destino.consultarCoordenada().fila();
		int destinoColumna = destino.consultarCoordenada().columna();
		
		// Si hay cambio en ambos ejes a la vez se trata de un movimiento diagonal no permitido, devuelve false
		if(origenFila != destinoFila && origenColumna != destinoColumna) {
			return false;
		}
		
		// Si no hay cambios en los ejes el movimiento es vertical u horizontal, luego, devuelve true
		return true;
	}
	
	/**
	 * Devuelve el sentido de la jugada
	 * 
	 * @return Sentido
	 */
	public Sentido consultarSentido() {
		// Toma las filas y columnas de origen y destino
		int origenFila = origen.consultarCoordenada().fila();
		int origenColumna = origen.consultarCoordenada().columna();
		
		int destinoFila = destino.consultarCoordenada().fila();
		int destinoColumna = destino.consultarCoordenada().columna();
		
		// Comprueba que no sea un movimiento en diagonal (no es un sentido valido)
		if(!esMovimientoHorizontalOVertical()) {
			return null;
		}
		
		// Comprueba los desplazamientos en coordenadas X,Y
		int variacionX = origenFila - destinoFila;
		int variacionY = origenColumna - destinoColumna;
		
		// Se declara la variable sentido
		Sentido sentido = null;
		
		// Si no hay variacion en Y entoncesn nos movemos en ese eje
		if(variacionY == 0) {
			// Dependiendo del valor de la variacion en X nos movemos hacia el norte o hacia el sur
			if(variacionX > 0) {
				sentido = Sentido.VERTICAL_N;
			}else{
				sentido = Sentido.VERTICAL_S;
			}
		}
		
		// Si no hay variacion en X entoncesn nos movemos en ese eje
		if(variacionX == 0) {
			// Dependiendo del valor de la variacion en Y nos movemos hacia el este u oeste
			if(variacionY > 0) {
				sentido = Sentido.HORIZONTAL_O;
			}else {
				sentido = Sentido.HORIZONTAL_E;
			}
		}
		
		// Devuelve el sentido
		return sentido;
	}
}