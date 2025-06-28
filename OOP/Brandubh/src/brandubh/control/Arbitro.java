package brandubh.control;

import brandubh.modelo.*;
import brandubh.util.*;

/**
 * Clase que implementa la logica del juego. Esta clase
 * controla el orden del juego y las interacciones entre los
 * distintos sistemas que componen el juego
 * 
 * @author <a href="azs1004@alu.ubu.es">Adrián Zamora Sánchez</a>
 * @see brandubh.modelo.Pieza
 * @see brandubh.modelo.Tablero
 * @see brandubh.modelo.Jugada
 * @see brandubh.textui.Brandubh
 * @version 1.0
 * @since 1.0
 * 
*/
public class Arbitro{
	/**
	 * Caracter que identifica los tipos de piezas
	 * 
	 * @see brandubh.modelo.Tablero
	 */
	private Tablero tablero;
	
	/**
	 * Caracter que identifica los tipos de piezas
	 * 
	 * @see brandubh.util.Color
	 */
	private Color turnoActual;
	
	/**
	 * Guarda la última jugada, útil para hacer comprobaciones
	 * 
	 * @see brandubh.modelo.Jugada
	 */
	private Jugada ultimaJugada;
	
	/**
	 * Caracter que identifica los tipos de piezas
	 * 
	 * @see java.lang.Object
	 */
	private int numeroJugadas;
	
	/**
	 * Constuctor de Arbitro, requiere de un tablero
	 * 
	 * @param tableroParam tablero pasado como parámetro
	 */
	public Arbitro(Tablero tableroParam) {
		// Establece el tablero
		tablero = tableroParam;
	}
	
	/**
	 * Método que devuelve el número de jugadas realizadas en este juego
	 * 
	 * @return int
	 */
	public int consultarNumeroJugada() {
		return numeroJugadas;
	}
	
	/**
	 * Método que coloca una serie de piezas en el tablero y establece el color con el
	 * que se juega en el siguiente turno
	 * 
	 * @param arrayTiposPiezas array con las piezas a colocar
	 * @param arrayCoordenada array con las coordenadas donde colocar las respectivas piezas
	 * @param turnoActual clor con el turno con el que comenzar
	 */
	public void colocarPiezas(TipoPieza[] arrayTiposPiezas, int[][] arrayCoordenada, Color turnoActual) {
		this.turnoActual = turnoActual;
		
		// Coloca en el tablero todas las piezas en sus coordenadas correspondientes
		for(int i = 0; i < arrayTiposPiezas.length; i++) {
			tablero.colocar(new Pieza(arrayTiposPiezas[i]), new Coordenada(arrayCoordenada[i][0], arrayCoordenada[i][1]));
		}
	}
	
	/**
	 * Método que coloca las piezas por defecto para inciar el juego
	 */
	public void colocarPiezasConfiguracionInicial() {
		// Coloca al rey
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(3,3));
		
		// Coloca los defensores
		tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(3,2));
		tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(3,4));
		tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(2,3));
		tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(4,3));
		
		// Coloca a los atacantes
		tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(3,0));
		tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(3,1));
		tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(3,6));
		tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(3,5));
		tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(0,3));
		tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(1,3));
		tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(6,3));
		tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(5,3));
		
		// Prepara el turno para comenzar a jugar con la configuración inicial
		turnoActual = Color.NEGRO;
	}
	
	/**
	 * Método que cambia el turno de atacantes a defensore o viceversa
	 * 
	 */
	public void cambiarTurno() {
		// Para los casos donde no se inciializa el tablero, se asume inicio en turno de atacantes
		if(turnoActual == null) {
			turnoActual = Color.NEGRO;
		}
		
		// Cambia al color contrario
		if(turnoActual == Color.NEGRO) {
			turnoActual = Color.BLANCO;
		}else {
			turnoActual = Color.NEGRO;
		}
	}
	
	/**
	 * Método que cambia el turno de atacantes a defensore o viceversa
	 * 
	 * @return Color
	 */
	public Color consultarTurno() {
		return turnoActual;
	}
	
	/**
	 * Método que devuelve el tablero sobre el que se está jugando
	 * 
	 * @return Tablero
	 */
	public Tablero consultarTablero() {
		return tablero;
	}
	
	/**
	 * Método que true si ha ganado el rey se encuentra en una provincia
	 * y false si no es así
	 * 
	 * @return boolean
	 */
	public boolean haGanadoRey() {
		Celda celdaRey;
		
		// Si no hay rey, no se puede obtener su ubicación en el tablero
		if(obtenerCoordenadasRey() == null) {
			return false;
		}
		
		// Se comprueba el tipo de celda donde se encuentra el rey
		celdaRey = tablero.consultarCelda(obtenerCoordenadasRey());
		if(celdaRey.consultarTipoCelda() == TipoCelda.PROVINCIA) {
			// Si es una provinacia, el rey ha ganado
			return true;
		}
		
		// En caso de no alcanzar una proviancia el rey aún no ha ganado
		return false;
	}
	
	/**
	 * Método que comprueba si el rey sigue en el tablero, en caso contrario ganan
	 * los atacantes
	 * 
	 * @return boolean
	 */
	public boolean haGanadoAtacante() {
		// Variable de control sobre el estado de la jugada anterior
		boolean reyEnUltimaJugada = false;
		
		// Si en la última jugada no han intervenido las piezas negras, el rey no se puede capturar
		if(ultimaJugada.consultarDestino().consultarColorDePieza() == Color.BLANCO) {
			return false;
		}
		
		// Comprueba todas las piezas cercanas al ultimo movimiento, si este no ha sido cerca del rey, no se debe
		// comprobar si cambia el atacante, evita el suicidio del rey si se coloca en una posicion insegura
		for(Celda celda : tablero.consultarCeldasContiguas(ultimaJugada.consultarDestino().consultarCoordenada())) {
			// Comprueba que no esta vacia la celda
			if(celda.estaVacia()) { continue; }
			
			// Comprueba si se ha visto afectado el rey
			if(celda.consultarPieza().consultarTipoPieza()== TipoPieza.REY) {
				// En caso afirmativo sale del bucle
				reyEnUltimaJugada = true;
				break;
			}
		}
		
		// Si el rey no se ve afectado por la ultima jugada deja de comprobar el estado del rey
		if(!reyEnUltimaJugada) {
			return false;
		}
		
		// Comprueba los distintos tipos de capturas sobre el rey
		if(comprobarCapturaReyEnTrono()) {
			return true;
		}else if(comprobarCapturaReyContiguoAlTrono()) {
			return true;
		}else if(comprobarCapturaReyNoContiguoAlTrono()) {
			return true;
		}
		
		return false;
	}
	
	/**
	 * Método que devuelve true si la jugada es legal (posicion valida y celda vacía)
	 * o false si no lo es
	 * 
	 * @param jugada jugada que se evalua
	 * @return boolean
	 */
	public boolean esMovimientoLegal(Jugada jugada) {
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
		
		// Incrementos en los ejes fila y columna, según el sentido del movimiento
		int filaIncremento = jugada.consultarSentido().consultarDesplazamientoEnFilas();
		int columnaIncremento = jugada.consultarSentido().consultarDesplazamientoEnColumnas();
		
		// Recorre todas las cendas entre el origen y el destino en la dirección de la jugada
		// en caso de encontrar alguna pieza, devuelve false, pues estas no pueden saltar a otras piezas
		// Recorre las celdas entre la posición desde la que se mueve hasta su ubicación final
		for (int filaInicial = origenFila + filaIncremento, columnaInicial = origenColumna + columnaIncremento;
		     (filaIncremento != 0 && filaInicial != destinoFila) || (columnaIncremento != 0 && columnaInicial != destinoColumna);
		     filaInicial += filaIncremento, columnaInicial += columnaIncremento) {
			
			// Comprueba si está vacía la celda
			if (!tablero.obtenerCelda(new Coordenada(filaInicial, columnaInicial)).estaVacia()) {
			    return false;
			}
		}
		
		// Si no se detecta ninguna ilegalidad entonces es un movimiento legar
		return true;
	}
	
	/**
	 * Método que desplaza las piezas en el tablero
	 * 
	 * @param jugada jugada que se realiza
	 */
	public void mover(Jugada jugada) {
		// Se manejan clones de celdas
		Celda celdaOrigen = jugada.consultarOrigen();
		Celda celdaDestino = jugada.consultarDestino();
		Pieza piezaOrigen = celdaOrigen.consultarPieza();
		
		// Se ubica la celda de destino en el tablero del arbitro y se coloca la pieza
		tablero.obtenerCelda(celdaDestino.consultarCoordenada()).colocar(piezaOrigen);
		
		// Se ubica la celda de origen en el tablero y se elimina la pieza en el origen
		tablero.obtenerCelda(celdaOrigen.consultarCoordenada()).eliminarPieza();
		
		// Aumenta el contador de jugadas
		numeroJugadas++;
		ultimaJugada = jugada;
	}
	
	/**
	 * Devuelve las coordenadas del rey
	 * 
	 * @return coordenada
	 */
	private Coordenada obtenerCoordenadasRey(){
		// Coordenada donde se encuentra el rey
		Coordenada coords = null;
		
		// Recorre todo el tablero hasta dar con el rey
		for(int i = 0; i < tablero.consultarNumeroFilas(); i++) {
			for(int j = 0; j < tablero.consultarNumeroColumnas(); j++) {
				if(!tablero.obtenerCelda(new Coordenada(i,j)).estaVacia()) {
					if(tablero.obtenerCelda(new Coordenada(i,j)).consultarPieza().consultarTipoPieza() == TipoPieza.REY) {
						coords = new Coordenada(i,j);
						return coords;
					}
				}
			}
		}
		
		// Devuelve las coordenadas
		return coords;
	}
	
	/**
	 * Si el rey no se ha movido en la última jugada se podrá capturar,
	 * esta comprobación evita que el rey se pueda "suicidar"
	 * 
	 * @return boolean
	 */
	private boolean posibleCapturaAlRey() {
		// Si el ultimo movimiento ha sido del rey, se devuelve null para que no se compruebe si está
		// o no en una situación de captura, pues el rey no se puede matar a si mismo
		if(ultimaJugada.consultarOrigen().consultarPieza() == null) {
			if(ultimaJugada.consultarDestino().consultarPieza().consultarTipoPieza() == TipoPieza.REY) {
				return false;
			}
		}else {
			if(ultimaJugada.consultarOrigen().consultarPieza().consultarTipoPieza() == TipoPieza.REY) {
				return false;
			}
		}
		
		return true;
	}
	
	/**
	 * Comprueba si el rey esta rodeado de piezas enemigas en el trono, en caso
	 * afirmativo elimina la pieza del rey
	 * 
	 * @return boolean
	 */
	private boolean comprobarCapturaReyEnTrono() {
		// Si el rey no se puede capturar se devuelve false
		if(!posibleCapturaAlRey()) {
			return false;
		}
		
		// Toma las coordenadas del rey
		Coordenada coordenadaRey = obtenerCoordenadasRey();
		if(coordenadaRey == null) {
			return false;
		}
		
		// Para capturas del rey estando en el trono
		if(tablero.consultarCelda(coordenadaRey).consultarTipoCelda() == TipoCelda.TRONO) {
			int atacantes = 0;
			
			// Se comprueba el número de atacantes en las celdas contiguas
			for(Celda celda : tablero.consultarCeldasContiguas(coordenadaRey)) {
				if(!celda.estaVacia()) {
					if(celda.consultarPieza().consultarTipoPieza().consultarColor() == Color.NEGRO) {
						atacantes++;
					}
				}else {
					// Si alguna celda esta vacía no se podrá capturar al rey, termina la comprobación
					return false;
				}
			}
			
			// Se captura al rey
			if(atacantes == 4) {
				return true;
			}
		}
		
		return false;
	}
	
	/**
	 * Comprueba si el rey esta rodeado de piezas por tres lados y el trono
	 * se encuentra en el lado restante, en caso afirmativo elimina la pieza del rey
	 * 
	 * @return boolean
	 */
	private boolean comprobarCapturaReyContiguoAlTrono() {
		// Si el rey no se puede capturar se devuelve false
		if(!posibleCapturaAlRey()) {
			return false;
		}
		
		// Se recoge la coordenada donde se encuentra el rey para comrpobar su situación	
		Coordenada coordenadaRey = obtenerCoordenadasRey();
		if(coordenadaRey == null) {
			return false;
		}
		
		// Para capturas entre el rey, el trono y tres atacantes
		int piezasAtacantes = 0;
		
		// Se recorren las celdas contiguas en busca de atacantes
		for(Celda celda : tablero.consultarCeldasContiguas(coordenadaRey)) {
			if(celda.consultarColorDePieza() == Color.NEGRO) {
				// Se incrementa el número de atacantes
				piezasAtacantes++;
			}
		}
		
		// Se vuelve a comprobar que al menos una celda adyacente sea el trono
		for(Celda celda : tablero.consultarCeldasContiguas(coordenadaRey)) {
			if(piezasAtacantes == 3 && celda.consultarTipoCelda() == TipoCelda.TRONO) {
				// Si hay tres atacantes y se encuentra adyacente al trono, será captura sobre el rey
				return true;
			}
		}
		return false;
	}
	
	/**
	 * Comprueba si el rey esta rodeado de piezas por tres lados y el trono
	 * se encuentra en el lado restante, en caso afirmativo elimina la pieza del rey
	 * 
	 * @return boolean
	 */
	private boolean comprobarCapturaReyNoContiguoAlTrono() {
		// Si el rey no se puede capturar se devuelve false
		if(!posibleCapturaAlRey()) {
			return false;
		}
		
		// Se recoge la coordenada donde se encuentra el rey para comrpobar su situación
		Coordenada coordenadaRey = obtenerCoordenadasRey();
		if(coordenadaRey == null) {
			return false;
		}
		
		// Condición para salir de la función si el rey está en el trono
		if(tablero.consultarCelda(coordenadaRey).consultarTipoCelda() == TipoCelda.TRONO){
			// Rey en el trono
			return false;
		}
		
		// Condición de exclusión para salir si el rey esta cerca del trono
		for(Celda celda : tablero.consultarCeldasContiguas(coordenadaRey)) {
			// Si no hay celda pasa a la siguiente iteración
			if(celda == null) {
				continue;
			}
			
			// Si el rey está contiguo al trono no se debe comprobar su captura en este método
			if(celda.consultarTipoCelda() == TipoCelda.TRONO) {
				return false;
			}
		}
		
		// Variables que controlan el atacaque sobre la pieza
		int piezasAtacantes = 0;
		int privinciasAdyacentes = 0;
		
		// Comprueba las capturas no adyacentes al trono en horizontal
		for(Celda celda : tablero.consultarCeldasContiguasEnHorizontal(coordenadaRey)) {
			// Si al menos una es null, no habrá captura
			if(celda == null) {
				break;
			}
			
			// Se comprueba si hay atacantes o una provincia contiguos
			if(!celda.estaVacia()) {
				if(celda.consultarColorDePieza() == Color.NEGRO) {
					piezasAtacantes++;
				}
			}else if(celda.consultarTipoCelda() == TipoCelda.PROVINCIA) {
				privinciasAdyacentes++;
			}
		}
		
		// Devuelve true, pues el rey podría ser capturado
		if(piezasAtacantes == 2) {
			return true;
		}else if(piezasAtacantes == 1 && privinciasAdyacentes == 1) {
			return true;
		}
		
		// Establece en 0 el numero de atacantes y provincias adyacentes
		piezasAtacantes = 0;
		privinciasAdyacentes = 0;
		
		// Comprueba las capturas no adyacentes al trono en vertical
		for(Celda celda : tablero.consultarCeldasContiguasEnVertical(coordenadaRey)) {
			// Si al menos una es null, no habrá captura
			if(celda == null) {
				break;
			}
			
			// Se comprueba si hay atacantes o una provincia contiguos
			if(!celda.estaVacia()) {
				if(celda.consultarColorDePieza() == Color.NEGRO) {
					piezasAtacantes++;
				}
			}else if(celda.consultarTipoCelda() == TipoCelda.PROVINCIA) {
				privinciasAdyacentes++;
			}
		}
		
		// Devuelve true, pues el rey podría ser capturado
		if(piezasAtacantes == 2) {
			return true;
		}else if(piezasAtacantes == 1 && privinciasAdyacentes == 1) {
			return true;
		}
		
		return false;
	}
	
	
	/**
	 * Método que controla las capturas de piezas
	 */
	public void realizarCapturasTrasMover() {
		// Se toma la coordenada a la que se desplaza una pieza en el último movimiento 
		Coordenada coordenadaUltimoMovimiento = ultimaJugada.consultarDestino().consultarCoordenada();
		
		// Variables que detectan las capturas
		int piezasAtacantes = 0;
		int provinciasAdyacentes = 0;
		
		// Recorre las celdas contiguas en horizontal a la ubicación del último movimiento
		for(Celda celdaContiguaHorizontal : tablero.consultarCeldasContiguasEnHorizontal(coordenadaUltimoMovimiento)) {
			// Si no hay celda salta la iteración
			if(celdaContiguaHorizontal == null) {
				continue;
			}
			
			// Salta la iteración, esta celda no contiene pieza que capturar
			if(celdaContiguaHorizontal.estaVacia()) {
				continue;
			}
			
			// Evita capturar al rey
			if(celdaContiguaHorizontal.consultarPieza().consultarTipoPieza() == TipoPieza.REY) {
				continue;
			}
			
			// Establece las variables que comprueban la captura a 0
			piezasAtacantes = 0;
			provinciasAdyacentes = 0;
			
			// Recorre cada celda contigua en horizontal, la siguiente en la misma dirección
			for(Celda celdaSiguienteHorizontal : tablero.consultarCeldasContiguasEnHorizontal(celdaContiguaHorizontal.consultarCoordenada())) {
				// Si no hay celda salta la iteración
				if(celdaSiguienteHorizontal == null) {
					continue;
				}
				 
				if(!celdaSiguienteHorizontal.estaVacia()) {
					// Si la celda no está vacía y es de un color distinto a la anterior, entonces está bajo ataque
					if(celdaContiguaHorizontal.consultarColorDePieza() != celdaSiguienteHorizontal.consultarColorDePieza()) {
						piezasAtacantes++;
					}
				}else if(celdaSiguienteHorizontal.consultarTipoCelda() == TipoCelda.PROVINCIA || celdaSiguienteHorizontal.consultarTipoCelda() == TipoCelda.TRONO) {
					// Si hay una celda de provincia o trono, se incrementa las provincias adyacentes
					provinciasAdyacentes++;
				}
				
				
				// Se comprueban los resultados para las celdas en vertical
				if(piezasAtacantes == 2) {
					// Si hay dos atacantes elimina la pieza
					tablero.eliminarPieza(celdaContiguaHorizontal.consultarCoordenada());
				}else if(piezasAtacantes == 1 && provinciasAdyacentes == 1) {
					// Si hay al menos un atacante y una provincia/trono también se captura la pieza
					tablero.eliminarPieza(celdaContiguaHorizontal.consultarCoordenada());
				}
			}
		}
		
		// Comprueba las celdas contiguas en vertical a la ubicación del último movimiento
		for(Celda celdaContiguaVertical : tablero.consultarCeldasContiguasEnVertical(coordenadaUltimoMovimiento)) {
			// Si no hay celda salta la iteración
			if(celdaContiguaVertical == null) {
				continue;
			}
			
			if(celdaContiguaVertical.estaVacia()) {
				// Saltamos iteración, esta celda no contiene pieza que capturar
				continue;
			}
			
			// Evita capturar al rey
			if(celdaContiguaVertical.consultarPieza().consultarTipoPieza() == TipoPieza.REY) {
				continue;
			}
			
			// Establece las variables que comprueban la captura a 0
			piezasAtacantes = 0;
			provinciasAdyacentes = 0;
			
			// Recorre cada celda contigua en vertical, la siguiente en la misma dirección
			for(Celda celdaSiguienteVertical : tablero.consultarCeldasContiguasEnVertical(celdaContiguaVertical.consultarCoordenada())) {
				// Si no hay celda salta la iteración
				if(celdaSiguienteVertical == null) {
					continue;
				}
				
				if(!celdaSiguienteVertical.estaVacia()) {
					// Si la celda no está vacía y es de un color distinto a la anterior, entonces está bajo ataque
					if(celdaContiguaVertical.consultarColorDePieza() != celdaSiguienteVertical.consultarColorDePieza()) {
						piezasAtacantes++;
					}
				}else if(celdaSiguienteVertical.consultarTipoCelda() == TipoCelda.PROVINCIA || celdaSiguienteVertical.consultarTipoCelda() == TipoCelda.TRONO) {
					// Si hay una celda de provincia o trono, se incrementa las provincias adyacentes
					provinciasAdyacentes++;
				}
				
				
				// Se comprueban los resultados para las celdas en vertical
				if(piezasAtacantes == 2) {
					// Si hay dos atacantes elimina la pieza
					tablero.eliminarPieza(celdaContiguaVertical.consultarCoordenada());
				}else if(piezasAtacantes == 1 && provinciasAdyacentes == 1) {
					// Si hay al menos un atacante y una provincia/trono también se captura la pieza
					tablero.eliminarPieza(celdaContiguaVertical.consultarCoordenada());
				}
			}
		}
	}
}