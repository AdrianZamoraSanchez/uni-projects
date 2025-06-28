package tafl.control;

import tafl.excepcion.CoordenadasIncorrectasException;
import tafl.modelo.Celda;
import tafl.modelo.Jugada;
import tafl.modelo.Pieza;
import tafl.modelo.Tablero;
import tafl.util.Color;
import tafl.util.Coordenada;
import tafl.util.TipoCelda;
import tafl.util.TipoPieza;

/**
 * Clase que implementa la logica del juego para partidas de ArdRi
 * 
 * @author <a href="azs1004@alu.ubu.es">Adrián Zamora Sánchez</a>
 * @see tafl.modelo.Pieza
 * @see tafl.modelo.Tablero
 * @see tafl.modelo.Jugada
 * @see tafl.textui.Tafl
 * @version 1.0
 * @since 1.0
 * 
*/
public class ArbitroArdRi extends ArbitroAbstracto{
	
	/**
	 * Constuctor del arbitro ArbitroArdRi, requiere de un tablero
	 * 
	 * @param tableroParam tablero pasado como parámetro
	 */
	public ArbitroArdRi(Tablero tableroParam) {
		super(tableroParam);
	}
	
	/**
	 * Método que coloca las piezas por defecto para inciar el juego
	 */
	public void colocarPiezasConfiguracionInicial(){
		try {
			// Coloca al rey
			tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(3,3));
			
			// Coloca los defensores
			tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(3,2));
			tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(3,4));
			tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(2,3));
			tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(4,3));
			tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(4,4));
			tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(4,2));
			tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(2,4));
			tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(2,2));
			
			// Coloca a los atacantes
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(6,4));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(6,2));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(4,6));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(2,6));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(4,0));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(2,0));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(0,4));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(0,2));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(3,0));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(3,1));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(3,6));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(3,5));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(0,3));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(1,3));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(6,3));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(5,3));
		} catch (CoordenadasIncorrectasException e) {
			e.printStackTrace();
		}
		
		// Prepara el turno para comenzar a jugar con la configuración inicial
		turnoActual = Color.NEGRO;
	}
	
	/**
	 * Método que devuelve true si la jugada es legal (posicion valida y celda vacía)
	 * o false si no lo es
	 * 
	 * @param jugada jugada que se evalua
	 * @return boolean
	 */
	public boolean esMovimientoLegal(Jugada jugada) {
		if(jugada == null) {
			throw new IllegalArgumentException();
		}
		
		// No se puede mover si no hay pieza que mover
		if(jugada.consultarOrigen().estaVacia()) {
			return false;
		}
		
		// Comprueba que no se mueva una pieza fuera de su turno
		if(jugada.consultarOrigen().consultarPieza().consultarTipoPieza().consultarColor() != turnoActual){
			return false;
		}
		
		// Si ya hay una pieza en la celda destino o el movimiento no es horizontal o vertical entonces es ilegal
		if(!jugada.esMovimientoHorizontalOVertical() || !jugada.consultarDestino().estaVacia()) {
			return false;
		}
		
		// Comprueba que solo un atacante no pueda moverse a una provincia
		if(jugada.consultarDestino().consultarTipoCelda() == TipoCelda.PROVINCIA && jugada.consultarOrigen().consultarPieza().consultarTipoPieza() != TipoPieza.REY) {
			return false;
		}
		
		// Solo el rey puede estar en el trono
		if(jugada.consultarOrigen().consultarPieza().consultarTipoPieza() != TipoPieza.REY && jugada.consultarDestino().consultarTipoCelda() == TipoCelda.TRONO) {
			return false;
		}
		
		// Toma las filas y columnas de origen y destino
		int origenFila = jugada.consultarOrigen().consultarCoordenada().fila();
		int origenColumna = jugada.consultarOrigen().consultarCoordenada().columna();
		
		int destinoFila = jugada.consultarDestino().consultarCoordenada().fila();
		int destinoColumna = jugada.consultarDestino().consultarCoordenada().columna();
		
		// No hay movimiento
		if((origenFila == destinoFila ) && (origenColumna == destinoColumna)) {
			return false;
		}
		
		// Comprueba que solo se muevan una casilla
		if(Math.abs(origenFila - destinoFila) > 1 || Math.abs(origenColumna - destinoColumna) > 1) {
			return false;
		}
		
		// Incrementos en los ejes fila y columna, según el sentido del movimiento
		int filaIncremento = jugada.consultarSentido().consultarDesplazamientoEnFilas();
		int columnaIncremento = jugada.consultarSentido().consultarDesplazamientoEnColumnas();

		// Recorre las celdas entre la posición desde la que se mueve hasta su ubicación final
		for (int filaInicial = origenFila + filaIncremento, columnaInicial = origenColumna + columnaIncremento;
		     (filaIncremento != 0 && filaInicial != destinoFila) || (columnaIncremento != 0 && columnaInicial != destinoColumna);
		     filaInicial += filaIncremento, columnaInicial += columnaIncremento) {
			
			// Comprueba si está vacía la celda
		    try{
		    	if (!tablero.obtenerCelda(new Coordenada(filaInicial, columnaInicial)).estaVacia()) { 
		    		return false;
			    }
		    }catch(CoordenadasIncorrectasException e) {
		    	e.printStackTrace();
		    }
		       
		}
		
		// Si no se detecta ninguna ilegalidad entonces es un movimiento legar
		return true;
	}
	
	/**
	 * Método que true si ha ganado el rey se encuentra en una provincia
	 * y false si no es así
	 * 
	 * @return boolean
	 */
	public boolean haGanadoRey() {
		Celda celdaRey = null;
		
		// Se comprueba el tipo de celda donde se encuentra el rey
		try {
			celdaRey = tablero.consultarCelda(obtenerCoordenadasRey());
		} catch (CoordenadasIncorrectasException e) {
			e.printStackTrace();
		}
		
		if(celdaRey.consultarCoordenada().fila() == 0 || celdaRey.consultarCoordenada().fila() == 6) {
			// Si es una provinacia, el rey ha ganado
			return true;
		}else if(celdaRey.consultarCoordenada().columna() == 0 || celdaRey.consultarCoordenada().columna() == 6) {
			// Si es una provinacia, el rey ha ganado
			return true;
		}
		
		// En caso de no alcanzar una proviancia el rey aún no ha ganado
		return false;
	}
}
