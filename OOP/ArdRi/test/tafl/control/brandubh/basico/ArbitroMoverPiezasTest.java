package tafl.control.brandubh.basico;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Timeout.ThreadMode.SEPARATE_THREAD;
import static tafl.control.TestUtil.fabricarJugada;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import tafl.control.ArbitroBrandubh;
import tafl.excepcion.CoordenadasIncorrectasException;
import tafl.modelo.Jugada;
import tafl.modelo.Tablero;
import tafl.util.Color;
import tafl.util.Coordenada;
import tafl.util.TipoPieza;

/**
 * Comprobación del ArbitroBrandubh de los movimientos de piezas sobre el tablero
 * trasladando las piezas.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 */
@DisplayName("Tests del ArbitroBrandubh sobre el movimiento de piezas.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public class ArbitroMoverPiezasTest extends tafl.control.basico.ArbitroAbstractoMoverPiezasTest {


	/** Generación del árbitro para testing. */
	// @formatter:off
	/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
	 * 7 - - - A - - -  
	 * 6 - - - A - - - 
	 * 5 - - - D - - - 
	 * 4 A A D R D A A 
	 * 3 - - - D - - -
	 * 2 - - - A - - - 
	 * 1 - - - A - - -
	 *   a b c d e f g   
	 */
	 // @formatter:on
	@BeforeEach
	void inicializar() {
		// Inyección de tablero para testing...
		tablero = new Tablero();
		arbitro = new ArbitroBrandubh(tablero);
		arbitro.colocarPiezasConfiguracionInicial();
	}	
	
	/**
	 * Comprobacion de mover pieza de atacante.
	 * 
     * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento de pieza del atacante en el turno inicial a distancia uno.")
	@Test
	void comprobarPiezaAtacanteACeldaCorrectaDistanciaUno()  throws CoordenadasIncorrectasException {
		Jugada jugada1 = fabricarJugada(tablero,0,3,0,2);
		arbitro.mover(jugada1);
		Jugada jugada2 = fabricarJugada(tablero,3,0,4,0);
		arbitro.mover(jugada2);
		Jugada jugada3 = fabricarJugada(tablero,5,3,5,4);
		arbitro.mover(jugada3);
		Jugada jugada4 = fabricarJugada(tablero,3,6,1,6);
		arbitro.mover(jugada4);
		assertAll("mover pieza de celda origen a destino",
				() -> assertThat("La pieza debería haberse movido en " + jugada1, arbitro.consultarTablero().consultarCelda(new Coordenada(0,3)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada1, arbitro.consultarTablero().consultarCelda(new Coordenada(0,2)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada2, arbitro.consultarTablero().consultarCelda(new Coordenada(3,0)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada2, arbitro.consultarTablero().consultarCelda(new Coordenada(4,0)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada3, arbitro.consultarTablero().consultarCelda(new Coordenada(5,3)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada3, arbitro.consultarTablero().consultarCelda(new Coordenada(5,4)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada4, arbitro.consultarTablero().consultarCelda(new Coordenada(3,6)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada4, arbitro.consultarTablero().consultarCelda(new Coordenada(1,6)).estaVacia(), is(false)),
				() -> assertThat("Debería haber 8 piezas atacantes",
						tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
				() -> assertThat("Debería haber 4 piezas defensoras",
						tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
				() -> assertThat("Debería haber rey", tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)),
				() -> assertThat("El turno no debería cambiar.", arbitro.consultarTurno(), is(Color.NEGRO)),
				() -> assertThat("El número de jugadas es incorrecto.", arbitro.consultarNumeroJugada(), is(4)));
	}
	

	/**
	 * Comprobacion de mover pieza de defensor.
	 * 
     * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento de pieza del defensor tras el turno inicial a distancia uno.")
	@Test
	void comprobarPiezaDefensorACeldaCorrectaDistanciaUno()  throws CoordenadasIncorrectasException {
		Jugada jugada1 = fabricarJugada(tablero,2,3,2,2);
		arbitro.mover(jugada1);
		Jugada jugada2 = fabricarJugada(tablero,3,2,4,2);
		arbitro.mover(jugada2);
		Jugada jugada3 = fabricarJugada(tablero,4,3,4,4);
		arbitro.mover(jugada3);
		Jugada jugada4 = fabricarJugada(tablero,3,4,2,4);
		arbitro.mover(jugada4);
		assertAll("mover pieza de celda origen a destino",
				() -> assertThat("La pieza debería haberse movido en " + jugada1, arbitro.consultarTablero().consultarCelda(new Coordenada(2,3)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada1, arbitro.consultarTablero().consultarCelda(new Coordenada(2,2)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada2, arbitro.consultarTablero().consultarCelda(new Coordenada(3,2)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada2, arbitro.consultarTablero().consultarCelda(new Coordenada(4,2)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada3, arbitro.consultarTablero().consultarCelda(new Coordenada(4,3)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada3, arbitro.consultarTablero().consultarCelda(new Coordenada(4,4)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada4, arbitro.consultarTablero().consultarCelda(new Coordenada(3,4)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada4, arbitro.consultarTablero().consultarCelda(new Coordenada(2,4)).estaVacia(), is(false)),
				() -> assertThat("Debería haber 8 piezas atacantes",
						tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
				() -> assertThat("Debería haber 4 piezas defensoras",
						tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
				() -> assertThat("Debería haber rey", tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)),
				() -> assertThat("El turno no debería cambiar.", arbitro.consultarTurno(), is(Color.NEGRO)),
				() -> assertThat("El número de jugadas es incorrecto.", arbitro.consultarNumeroJugada(), is(4)));
	}
	
	
	/**
	 * Comprobacion de mover pieza de atacante.
	 * 
     * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento de pieza del atacante en el turno inicial a distancia dos.")
	@Test
	void comprobarPiezaAtacanteACeldaCorrectaDistanciaDos()  throws CoordenadasIncorrectasException {
		Jugada jugada1 = fabricarJugada(tablero,0,3,0,1);
		arbitro.mover(jugada1);
		Jugada jugada2 = fabricarJugada(tablero,3,0,5,0);
		arbitro.mover(jugada2);
		Jugada jugada3 = fabricarJugada(tablero,5,3,5,5);
		arbitro.mover(jugada3);
		Jugada jugada4 = fabricarJugada(tablero,3,6,1,6);
		arbitro.mover(jugada4);
		assertAll("mover pieza de celda origen a destino",
				() -> assertThat("La pieza debería haberse movido en " + jugada1, arbitro.consultarTablero().consultarCelda(new Coordenada(0,3)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada1, arbitro.consultarTablero().consultarCelda(new Coordenada(0,1)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada2, arbitro.consultarTablero().consultarCelda(new Coordenada(3,0)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada2, arbitro.consultarTablero().consultarCelda(new Coordenada(5,0)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada3, arbitro.consultarTablero().consultarCelda(new Coordenada(5,3)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada3, arbitro.consultarTablero().consultarCelda(new Coordenada(5,5)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada4, arbitro.consultarTablero().consultarCelda(new Coordenada(3,6)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada4, arbitro.consultarTablero().consultarCelda(new Coordenada(1,6)).estaVacia(), is(false)),
				() -> assertThat("Debería haber 8 piezas atacantes",
						tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
				() -> assertThat("Debería haber 4 piezas defensoras",
						tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
				() -> assertThat("Debería haber rey", tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)),
				() -> assertThat("El turno no debería cambiar.", arbitro.consultarTurno(), is(Color.NEGRO)),
				() -> assertThat("El número de jugadas es incorrecto.", arbitro.consultarNumeroJugada(), is(4)));
	}
	

	/**
	 * Comprobacion de mover pieza de defensor.
	 * 
     * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento de pieza del defensor tras el turno inicial a distancia dos.")
	@Test
	void comprobarPiezaDefensorACeldaCorrectaDistanciaDos()  throws CoordenadasIncorrectasException {
		Jugada jugada1 = fabricarJugada(tablero,2,3,2,1);
		arbitro.mover(jugada1);
		Jugada jugada2 = fabricarJugada(tablero,3,2,5,2);
		arbitro.mover(jugada2);
		Jugada jugada3 = fabricarJugada(tablero,4,3,4,5);
		arbitro.mover(jugada3);
		Jugada jugada4 = fabricarJugada(tablero,3,4,1,4);
		arbitro.mover(jugada4);
		assertAll("mover pieza de celda origen a destino",
				() -> assertThat("La pieza debería haberse movido en " + jugada1, arbitro.consultarTablero().consultarCelda(new Coordenada(2,3)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada1, arbitro.consultarTablero().consultarCelda(new Coordenada(2,1)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada2, arbitro.consultarTablero().consultarCelda(new Coordenada(3,2)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada2, arbitro.consultarTablero().consultarCelda(new Coordenada(5,2)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada3, arbitro.consultarTablero().consultarCelda(new Coordenada(4,3)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada3, arbitro.consultarTablero().consultarCelda(new Coordenada(4,5)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada4, arbitro.consultarTablero().consultarCelda(new Coordenada(3,4)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada4, arbitro.consultarTablero().consultarCelda(new Coordenada(1,4)).estaVacia(), is(false)),
				() -> assertThat("Debería haber 8 piezas atacantes",
						tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
				() -> assertThat("Debería haber 4 piezas defensoras",
						tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
				() -> assertThat("Debería haber rey", tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)),
				() -> assertThat("El turno no debería cambiar.", arbitro.consultarTurno(), is(Color.NEGRO)),
				() -> assertThat("El número de jugadas es incorrecto.", arbitro.consultarNumeroJugada(), is(4)));
	}
	
	
	/**
	 * Comprobacion de mover pieza de atacante a distancia tres.
	 * 
     * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento de pieza del atacante en el turno inicial a distancia tres.")
	@Test
	void comprobarPiezaAtacanteACeldaDistanciaTres()  throws CoordenadasIncorrectasException {
		Jugada jugada1 = fabricarJugada(tablero,0,3,0,0);
		arbitro.mover(jugada1);
		Jugada jugada2 = fabricarJugada(tablero,3,0,6,0);
		arbitro.mover(jugada2);
		Jugada jugada3 = fabricarJugada(tablero,5,3,5,0);
		arbitro.mover(jugada3);
		Jugada jugada4 = fabricarJugada(tablero,3,6,0,6);
		arbitro.mover(jugada4);
		assertAll("mover pieza de celda origen a destino",
				() -> assertThat("La pieza debería haberse movido en " + jugada1, arbitro.consultarTablero().consultarCelda(new Coordenada(0,3)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada1, arbitro.consultarTablero().consultarCelda(new Coordenada(0,0)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada2, arbitro.consultarTablero().consultarCelda(new Coordenada(3,0)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada2, arbitro.consultarTablero().consultarCelda(new Coordenada(6,0)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada3, arbitro.consultarTablero().consultarCelda(new Coordenada(5,3)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada3, arbitro.consultarTablero().consultarCelda(new Coordenada(5,0)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada4, arbitro.consultarTablero().consultarCelda(new Coordenada(3,6)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada4, arbitro.consultarTablero().consultarCelda(new Coordenada(0,6)).estaVacia(), is(false)),
				() -> assertThat("Debería haber 8 piezas atacantes",
						tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
				() -> assertThat("Debería haber 4 piezas defensoras",
						tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
				() -> assertThat("Debería haber rey", tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)),
				() -> assertThat("El turno no debería cambiar.", arbitro.consultarTurno(), is(Color.NEGRO)),
				() -> assertThat("El número de jugadas es incorrecto.", arbitro.consultarNumeroJugada(), is(4)));
	}
	

	/**
	 * Comprobacion de mover pieza de defensor a distancia tres.
	 * 
     * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento de pieza del defensor tras el turno inicial a distancia tres.")
	@Test
	void comprobarPiezaDefensorACeldaDistanciaTres()  throws CoordenadasIncorrectasException {
		Jugada jugada1 = fabricarJugada(tablero,2,3,2,0);
		arbitro.mover(jugada1);
		Jugada jugada2 = fabricarJugada(tablero,3,2,6,2);
		arbitro.mover(jugada2);
		Jugada jugada3 = fabricarJugada(tablero,4,3,4,6);
		arbitro.mover(jugada3);
		Jugada jugada4 = fabricarJugada(tablero,3,4,0,4);
		arbitro.mover(jugada4);
		assertAll("mover pieza de celda origen a destino",
				() -> assertThat("La pieza debería haberse movido en " + jugada1, arbitro.consultarTablero().consultarCelda(new Coordenada(2,3)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada1, arbitro.consultarTablero().consultarCelda(new Coordenada(2,0)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada2, arbitro.consultarTablero().consultarCelda(new Coordenada(3,2)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada2, arbitro.consultarTablero().consultarCelda(new Coordenada(6,2)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada3, arbitro.consultarTablero().consultarCelda(new Coordenada(4,3)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada3, arbitro.consultarTablero().consultarCelda(new Coordenada(4,6)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada4, arbitro.consultarTablero().consultarCelda(new Coordenada(3,4)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada4, arbitro.consultarTablero().consultarCelda(new Coordenada(0,4)).estaVacia(), is(false)),
				() -> assertThat("Debería haber 8 piezas atacantes",
						tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
				() -> assertThat("Debería haber 4 piezas defensoras",
						tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
				() -> assertThat("Debería haber rey", tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)),
				() -> assertThat("El turno no debería cambiar.", arbitro.consultarTurno(), is(Color.NEGRO)),
				() -> assertThat("El número de jugadas es incorrecto.", arbitro.consultarNumeroJugada(), is(4)));
	}
}
