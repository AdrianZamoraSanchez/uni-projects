package tafl.control.brandubh.avanzado;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Timeout.ThreadMode.SEPARATE_THREAD;
import static tafl.control.TestUtil.fabricarJugada;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import tafl.control.ArbitroBrandubh;
import tafl.excepcion.CoordenadasIncorrectasException;
import tafl.modelo.Tablero;
import tafl.util.Color;
import tafl.util.TipoPieza;

/**
 * Partida completa con ArbitroBrandubh y finalización con victoria
 * de atacante o defensor, aplicando además alguna captura.
 * 
 * Demuestra la posibilidad de jugar una partida completa 
 * incluyendo además la captura de piezas.
 *  
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@DisplayName("Tests del ArbitroBrandubh sobre victorias avanzadas completas.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public class ArbitroPartidaCompletaTest extends tafl.control.avanzado.ArbitroAbstractoPartidaCompletaTest {

	/** Generación del árbitro para testing. */
	@BeforeEach
	void inicializar() {
		// Inyección de tablero para testing...
		tablero = new Tablero();
		arbitro = new ArbitroBrandubh(tablero);
		arbitro.colocarPiezasConfiguracionInicial();
	}
	
	/**
	 * Comprueba una partida con victoria del defensor con una captura de pieza del defensor.
	 * 
     * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 */
	@Test
	@DisplayName("Comprueba que hay victoria del defensor con una captura de pieza del defensor.")
	void probarPartidaPartidaConVictoriaDefensorConUnaCaptura() throws CoordenadasIncorrectasException {

		// given
		arbitro.mover(fabricarJugada(tablero, 3, 5, 5, 5)); // atacante f4f2
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 4, 6, 4)); // defensor e4e1
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 5, 5, 6, 5)); // atacante f2f1
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 3, 3, 5)); // rey mueve d4f4
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 6, 5, 4, 5)); // atacante f1f3
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 5, 0, 5)); // rey f4f7
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 4, 5, 1, 5)); // atacante f3f6
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();

		// when
		arbitro.mover(fabricarJugada(tablero, 0, 5, 0, 6)); // f7g7 rey llega a provincia y gana
		// no se realizan capturas ni se cambia el turno
		// para observar el estado de la partida en el test

		// @formatter:off
		/* El estado del tablero debería ser:
		 * 7 - - - A - - R  
		 * 6 - - - A - A -
		 * 5 - - - D - - - 
		 * 4 A A D - - - A 
		 * 3 - - - D - - -
		 * 2 - - - A - - - 
		 * 1 - - - A - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on

		// then
		final String cadenaEsperada = """
				7 - - - A - - R
				6 - - - A - A -
				5 - - - D - - -
				4 A A D - - - A
				3 - - - D - - -
				2 - - - A - - -
				1 - - - A - - -
				  a b c d e f g """.replaceAll("\\s", "");
		String cadenaObtenida = arbitro.consultarTablero().aTexto().replaceAll("\\s", "");
		
		assertAll("Victoria del defensor alcanzando provincia.",
				
				() -> assertThat("No debería ganar el atacante.", arbitro.haGanadoAtacante(), is(false)),
				() -> assertThat("Debería ganar el defensor.", arbitro.haGanadoRey(), is(true)),
				() -> assertThat("Deberían estar todas las piezas atacantes sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
				() -> assertThat("Deberían estar solo tres defensores sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(3)),
				() -> assertThat("Deberían estar todavía el rey sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)),
				() -> assertThat("La partida se desarrolla en un número erróneo de jugadas.",
						arbitro.consultarNumeroJugada(), is(8)),
				() -> assertThat("El turno debería ser del jugado defensor.", arbitro.consultarTurno(),
						is(Color.BLANCO)),
				() -> assertEquals(cadenaEsperada, cadenaObtenida, "Estado del tablero en formato texto incorrecto")
				);
	}

	/**
	 * Comprueba una partida con victoria del atacante con una doble captura de piezas del defensor.
	 * 
     * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 */
	@Test
	@DisplayName("Comprueba que hay victoria del atacante con una doble captura de piezas del defensor.")
	void probarPartidaPartidaConVictoriaAtacanteCapturandoReyAMitadDeTableroConDobleCapturaDePiezasDelDefensor() throws CoordenadasIncorrectasException {

		// given
		arbitro.mover(fabricarJugada(tablero, 3, 5, 2, 5)); // atacante f4f5
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 4, 3, 4, 5)); // defensor d3f3
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 5, 3, 5, 5)); // atacante d2f2
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 3, 4, 3)); // defensor d4d3 mueve el rey
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 2, 5, 3, 5)); // atacante f5f4 realiza doble captura de piezas del defensor utilizando el trono
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 4, 3, 5, 3)); // defensor d3d2 mueve el rey
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 1, 5, 1)); // atacante b4b2
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 5, 3, 5, 2)); // defensor d2c2 mueve el rey
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();		
		// when
		arbitro.mover(fabricarJugada(tablero, 5, 5, 5, 3)); // atacante f2d2 captura el rey
		// no se realizan capturas ni se cambia el turno
		// para observar el estado de la partida en el test

		// @formatter:off
		/* El estado del tablero debería ser:
		 * 7 - - - A - - -  
		 * 6 - - - A - - -
		 * 5 - - - D - - - 
		 * 4 A - D - - A A 
		 * 3 - - - - - - -
		 * 2 - A R A - - - 
		 * 1 - - - A - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on

		// then
		final String cadenaEsperada = """
				7 - - - A - - -
				6 - - - A - - -
				5 - - - D - - -
				4 A - D - - A A
				3 - - - - - - -
				2 - A R A - - -
				1 - - - A - - -
				  a b c d e f g """.replaceAll("\\s", "");
		String cadenaObtenida = arbitro.consultarTablero().aTexto().replaceAll("\\s", "");
		
		assertAll("Victoria de atacante rodeando al rey con dos atacantes.",
				
				() -> assertThat("Debería ganar el atacante.", arbitro.haGanadoAtacante(), is(true)),
				() -> assertThat("No debería ganar el defensor.", arbitro.haGanadoRey(), is(false)),
				() -> assertThat("Deberían estar todas las piezas atacantes sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
				() -> assertThat("Deberían estar solo dos defensores sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(2)),
				() -> assertThat("Deberían estar todavía el rey sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)),
				() -> assertThat("La partida se desarrolla en un número erróneo de jugadas.",
						arbitro.consultarNumeroJugada(), is(9)),
				() -> assertThat("El turno debería ser del jugado atacante.", arbitro.consultarTurno(),
						is(Color.NEGRO)),
				() -> assertEquals(cadenaEsperada, cadenaObtenida, "Estado del tablero en formato texto incorrecto")
				);
	}
}
