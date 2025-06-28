package brandubh.control.basico;

import static brandubh.control.TestUtil.fabricarJugada;
import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertAll;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import brandubh.control.Arbitro;
import brandubh.modelo.Jugada;
import brandubh.modelo.Tablero;
import brandubh.util.Color;
import brandubh.util.Coordenada;

/**
 * Comprobación de los movimientos de piezas sobre el tablero
 * trasladando las piezas.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 */
@DisplayName("Tests del Arbitro sobre el movimiento de piezas.")
@Timeout(value = 2, unit = TimeUnit.SECONDS) // Time out global para todos los tests salvo los de ciclo de vida
public class ArbitroMoverPiezasTest {
	
	/** Árbitro de testing. */
	private Arbitro arbitro;

	/** Tablero de testing. */
	private Tablero tablero;

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
		arbitro = new Arbitro(tablero);
		arbitro.colocarPiezasConfiguracionInicial();
	}
	
	/**
	 * Comprobacion de mover pieza de atacante.
	 * 
	 * @see brandubh.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento de pieza del atacante en el turno inicial a distancia uno.")
	@Test
	void comprobarPiezaAtacanteACeldaCorrectaDistanciaUno() {
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
				() -> assertThat("El turno no debería cambiar.", arbitro.consultarTurno(), is(Color.NEGRO)),
				() -> assertThat("El número de jugadas es incorrecto.", arbitro.consultarNumeroJugada(), is(4)));
	}
	

	/**
	 * Comprobacion de mover pieza de defensor.
	 * 
	 * @see brandubh.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento de pieza del defensor tras el turno inicial a distancia uno.")
	@Test
	void comprobarPiezaDefensorACeldaCorrectaDistanciaUno() {
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
				() -> assertThat("El turno no debería cambiar.", arbitro.consultarTurno(), is(Color.NEGRO)),
				() -> assertThat("El número de jugadas es incorrecto.", arbitro.consultarNumeroJugada(), is(4)));
	}
	
	/**
	 * Comprobacion de mover pieza de atacante.
	 * 
	 * @see  brandubh.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento de pieza del atacante en el turno inicial a distancia dos.")
	@Test
	void comprobarPiezaAtacanteACeldaCorrectaDistanciaDos() {
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
				() -> assertThat("El turno no debería cambiar.", arbitro.consultarTurno(), is(Color.NEGRO)),
				() -> assertThat("El número de jugadas es incorrecto.", arbitro.consultarNumeroJugada(), is(4)));
	}
	

	/**
	 * Comprobacion de mover pieza de defensor.
	 * 
	 * @see brandubh.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento de pieza del defensor tras el turno inicial a distancia dos.")
	@Test
	void comprobarPiezaDefensorACeldaCorrectaDistanciaDos() {
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
				() -> assertThat("El turno no debería cambiar.", arbitro.consultarTurno(), is(Color.NEGRO)),
				() -> assertThat("El número de jugadas es incorrecto.", arbitro.consultarNumeroJugada(), is(4)));
	}
	
	
	/**
	 * Comprobacion de mover pieza de atacante a distancia tres.
	 * 
	 * @see brandubh.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento de pieza del atacante en el turno inicial a distancia tres.")
	@Test
	void comprobarPiezaAtacanteACeldaDistanciaTres() {
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
				() -> assertThat("El turno no debería cambiar.", arbitro.consultarTurno(), is(Color.NEGRO)),
				() -> assertThat("El número de jugadas es incorrecto.", arbitro.consultarNumeroJugada(), is(4)));
	}
	

	/**
	 * Comprobacion de mover pieza de defensor a distancia tres.
	 * 
	 * @see brandubh.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento de pieza del defensor tras el turno inicial a distancia tres.")
	@Test
	void comprobarPiezaDefensorACeldaDistanciaTres() {
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
				() -> assertThat("El turno no debería cambiar.", arbitro.consultarTurno(), is(Color.NEGRO)),
				() -> assertThat("El número de jugadas es incorrecto.", arbitro.consultarNumeroJugada(), is(4)));
	}
}
