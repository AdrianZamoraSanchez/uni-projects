package tafl.control.ardri.basico;

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

import tafl.control.ArbitroArdRi;
import tafl.excepcion.CoordenadasIncorrectasException;
import tafl.modelo.Jugada;
import tafl.modelo.Tablero;
import tafl.util.Color;
import tafl.util.Coordenada;
import tafl.util.TipoPieza;

/**
 * Comprobación del ArbitroArdRi de los movimientos de piezas sobre el tablero
 * trasladando las piezas.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 */
@DisplayName("Tests del ArbitroArdRi sobre el movimiento de piezas.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public class ArbitroMoverPiezasTest extends tafl.control.basico.ArbitroAbstractoMoverPiezasTest {

	/** Generación del árbitro para testing. */
	// @formatter:off
	/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
	 * 7 - - A A A - -  
	 * 6 - - - A - - - 
	 * 5 A - D D D - A 
	 * 4 A A D R D A A 
	 * 3 A - D D D - A
	 * 2 - - - A - - - 
	 * 1 - - A A A - -
	 *   a b c d e f g   
	 */
	 // @formatter:on
	@BeforeEach
	void inicializar() {
		// Inyección de tablero para testing...
		tablero = new Tablero();
		arbitro = new ArbitroArdRi(tablero);
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
		Jugada jugada1 = fabricarJugada(tablero,0,2,0,1);
		arbitro.mover(jugada1);
		Jugada jugada2 = fabricarJugada(tablero,3,1,4,1);
		arbitro.mover(jugada2);
		Jugada jugada3 = fabricarJugada(tablero,5,3,5,4);
		arbitro.mover(jugada3);
		Jugada jugada4 = fabricarJugada(tablero,3,5,2,5);
		arbitro.mover(jugada4);
		assertAll("mover pieza de celda origen a destino",
				() -> assertThat("La pieza debería haberse movido en " + jugada1, arbitro.consultarTablero().consultarCelda(new Coordenada(0,2)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada1, arbitro.consultarTablero().consultarCelda(new Coordenada(0,1)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada2, arbitro.consultarTablero().consultarCelda(new Coordenada(3,1)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada2, arbitro.consultarTablero().consultarCelda(new Coordenada(4,1)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada3, arbitro.consultarTablero().consultarCelda(new Coordenada(5,3)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada3, arbitro.consultarTablero().consultarCelda(new Coordenada(5,4)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada4, arbitro.consultarTablero().consultarCelda(new Coordenada(3,5)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada4, arbitro.consultarTablero().consultarCelda(new Coordenada(2,5)).estaVacia(), is(false)),
				() -> assertThat("Debería haber 16 piezas atacantes",
						tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(16)),
				() -> assertThat("Debería haber 8 piezas defensoras",
						tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(8)),
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
		Jugada jugada1 = fabricarJugada(tablero,2,2,2,1);
		arbitro.mover(jugada1);
		Jugada jugada2 = fabricarJugada(tablero,4,2,5,2);
		arbitro.mover(jugada2);
		Jugada jugada3 = fabricarJugada(tablero,4,4,4,5);
		arbitro.mover(jugada3);
		Jugada jugada4 = fabricarJugada(tablero,2,4,1,4);
		arbitro.mover(jugada4);
		assertAll("mover pieza de celda origen a destino",
				() -> assertThat("La pieza debería haberse movido en " + jugada1, arbitro.consultarTablero().consultarCelda(new Coordenada(2,2)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada1, arbitro.consultarTablero().consultarCelda(new Coordenada(2,1)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada2, arbitro.consultarTablero().consultarCelda(new Coordenada(4,2)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada2, arbitro.consultarTablero().consultarCelda(new Coordenada(5,2)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada3, arbitro.consultarTablero().consultarCelda(new Coordenada(4,4)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada3, arbitro.consultarTablero().consultarCelda(new Coordenada(4,5)).estaVacia(), is(false)),
				() -> assertThat("La pieza debería haberse movido en " + jugada4, arbitro.consultarTablero().consultarCelda(new Coordenada(2,4)).estaVacia(), is(true)),
				() -> assertThat("La pieza debería haberse movido en " + jugada4, arbitro.consultarTablero().consultarCelda(new Coordenada(1,4)).estaVacia(), is(false)),
				() -> assertThat("Debería haber 16 piezas atacantes",
						tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(16)),
				() -> assertThat("Debería haber 8 piezas defensoras",
						tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(8)),
				() -> assertThat("Debería haber rey", tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)),
				() -> assertThat("El turno no debería cambiar.", arbitro.consultarTurno(), is(Color.NEGRO)),
				() -> assertThat("El número de jugadas es incorrecto.", arbitro.consultarNumeroJugada(), is(4)));
	}
}
