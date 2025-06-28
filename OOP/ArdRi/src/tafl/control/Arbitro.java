package tafl.control;

import tafl.excepcion.CoordenadasIncorrectasException;
import tafl.modelo.Jugada;
import tafl.modelo.Tablero;
import tafl.util.Color;
import tafl.util.Coordenada;
import tafl.util.TipoPieza;

/**
 * Interfaz que implementa los metodos de los arbitros
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
public interface Arbitro{
	
	/**
	 * Método que devuelve el número de jugadas realizadas en este juego
	 * 
	 * @return int
	 */
	int consultarNumeroJugada();
	
	/**
	 * Método que coloca una serie de piezas en el tablero y establece el color con el
	 * que se juega en el siguiente turno
	 * 
	 * @param arrayTiposPiezas array con las piezas a colocar
	 * @param arrayCoordenada array con las coordenadas donde colocar las respectivas piezas
	 * @param turnoActual clor con el turno con el que comenzar
	 * @throws CoordenadasIncorrectasException excepción de coordenadas incorrectas
	 */
	void colocarPiezas(TipoPieza[] arrayTiposPiezas, int[][] arrayCoordenada, Color turnoActual) throws CoordenadasIncorrectasException;
	
	/**
	 * Método que coloca las piezas por defecto para inciar el juego
	 */
	void colocarPiezasConfiguracionInicial();
	
	/**
	 * Método que cambia el turno de atacantes a defensore o viceversa
	 * 
	 */
	void cambiarTurno();
	
	/**
	 * Método que cambia el turno de atacantes a defensore o viceversa
	 * 
	 * @return Color
	 */
	Color consultarTurno();
	
	/**
	 * Método que devuelve el tablero sobre el que se está jugando
	 * 
	 * @return Tablero
	 */
	Tablero consultarTablero();
	
	/**
	 * Método que true si ha ganado el rey se encuentra en una provincia
	 * y false si no es así
	 * 
	 * @return boolean
	 */
	boolean haGanadoRey();
	
	/**
	 * Método que comprueba si el rey sigue en el tablero, en caso contrario ganan
	 * los atacantes
	 * 
	 * @return boolean
	 */
	boolean haGanadoAtacante();
	
	/**
	 * Método que devuelve true si la jugada es legal (posicion valida y celda vacía)
	 * o false si no lo es
	 * 
	 * @param jugada jugada que se evalua
	 * @return boolean
	 */
	boolean esMovimientoLegal(Jugada jugada);
	
	/**
	 * Método que desplaza las piezas en el tablero
	 * 
	 * @param jugada jugada que se realiza
	 * @throws CoordenadasIncorrectasException excepción por coordenadas incorrectas
	 */
	void mover(Jugada jugada) throws CoordenadasIncorrectasException;
	
	/**
	 * Devuelve las coordenadas del rey
	 * 
	 * @return coordenada
	 * @throws CoordenadasIncorrectasException excepción de coordenadas incorrectas
	 */
	Coordenada obtenerCoordenadasRey() throws CoordenadasIncorrectasException;
	
	/**
	 * Si el rey no se ha movido en la última jugada se podrá capturar,
	 * esta comprobación evita que el rey se pueda "suicidar"
	 * 
	 * @return boolean
	 */
	boolean posibleCapturaAlRey();
	
	/**
	 * Comprueba si el rey esta rodeado de piezas enemigas en el trono, en caso
	 * afirmativo elimina la pieza del rey
	 * 
	 * @return boolean
	 * @throws CoordenadasIncorrectasException excepción de coordenadas incorrectas
	 */
	boolean comprobarCapturaReyEnTrono() throws CoordenadasIncorrectasException;
	
	/**
	 * Comprueba si el rey esta rodeado de piezas por tres lados y el trono
	 * se encuentra en el lado restante, en caso afirmativo elimina la pieza del rey
	 * 
	 * @return boolean
	 * @throws CoordenadasIncorrectasException excepción de coordenadas incorrectas
	 */
	boolean comprobarCapturaReyContiguoAlTrono() throws CoordenadasIncorrectasException;
	
	/**
	 * Comprueba si el rey esta rodeado de piezas por tres lados y el trono
	 * se encuentra en el lado restante, en caso afirmativo elimina la pieza del rey
	 * 
	 * @return boolean
	 * @throws CoordenadasIncorrectasException excepción de coordenadas incorrectas
	 */
	boolean comprobarCapturaReyNoContiguoAlTrono() throws CoordenadasIncorrectasException;
	
	/**
	 * Método que controla las capturas de piezas
	 * 
	 * @throws CoordenadasIncorrectasException excepción de coordenadas incorrectas
	 */
	void realizarCapturasTrasMover() throws CoordenadasIncorrectasException;
}
