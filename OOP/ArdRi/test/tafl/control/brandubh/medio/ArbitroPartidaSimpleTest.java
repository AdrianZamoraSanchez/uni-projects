package tafl.control.brandubh.medio;

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
 * Partida simple del ArbitroBrandubh con pocos movimientos y finalización con victoria
 * de atacante o defensor, sin aplicar ninguna captura.
 * Tampoco se comprueba la legalidad de las jugadas.
 * 
 * Demuestra la posibilidad de jugar una partida mínima sin incluir
 * la funcionalidad de capturas ni comprobación de legalidad.
 *  
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@DisplayName("Tests del ArbitroBrandubh sobre partidas simples.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public class ArbitroPartidaSimpleTest extends tafl.control.medio.ArbitroAbstractoPartidaSimpleTest {

	/** Generación del árbitro para testing. */
	@BeforeEach
	void inicializar() {
		// Inyección de tablero para testing...
		tablero = new Tablero();
		arbitro = new ArbitroBrandubh(tablero);
		arbitro.colocarPiezasConfiguracionInicial();
	}

	/**
	 * Comprueba una partida simple con victoria del defensor.
	 * 
     * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 */
	@Test
	@DisplayName("Comprueba que hay victoria del defensor en una partida simple sin capturas.")
	void probarPartidaSimpleConVictoriaDelDefensor() throws CoordenadasIncorrectasException {

		// given
		arbitro.mover(fabricarJugada(tablero, 3, 5, 0, 5)); // atacante
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 4, 6, 4)); // defensor
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 6, 2, 6)); // atacante
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 3, 3, 6)); // rey
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 2, 6, 1, 6)); // atacante
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();

		// @formatter:off
		/* El estado del tablero antes del último movimiento debería ser:
		 * 7 - - - A - A -  
		 * 6 - - - A - - A 
		 * 5 - - - D - - - 
		 * 4 A A D - - - R 
		 * 3 - - - D - - -
		 * 2 - - - A - - - 
		 * 1 - - - A D - -
		 *   a b c d e f g   
		 */
		 // @formatter:on

		// when
		arbitro.mover(fabricarJugada(tablero, 3, 6, 6, 6)); // rey alcanza provincia
		// then
		final String cadenaEsperada = """
				7 - - - A - A -
				6 - - - A - - A
				5 - - - D - - -
				4 A A D - - - -
				3 - - - D - - -
				2 - - - A - - -
				1 - - - A D - R
				  a b c d e f g """.replaceAll("\\s", "");
		String cadenaObtenida = arbitro.consultarTablero().aTexto().replaceAll("\\s", "");
		assertAll("Victoria del rey alcanzando provincia en partida basica.",
				() -> assertThat("Debería ganar el defensor.", arbitro.haGanadoRey(), is(true)),
				() -> assertThat("No debería ganar el atacante.", arbitro.haGanadoAtacante(), is(false)),
				() -> assertThat("Deberían estar todas las piezas atacantes sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
				() -> assertThat("Deberían estar todas las piezas defensoras sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
				() -> assertThat("Deberían estar todavía el rey sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)),
				() -> assertThat("La partida se desarrolla en un número erróneo de jugadas.",
						arbitro.consultarNumeroJugada(), is(6)),
				() -> assertThat("El turno debería ser del jugado defensor.", arbitro.consultarTurno(),
						is(Color.BLANCO)),
				() -> assertEquals(cadenaEsperada, cadenaObtenida, "Estado del tablero en formato texto incorrecto")
				);

	}

	/**
	 * Comprueba una partida simple con victoria del atacante.
	 * 
     * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 */
	@Test
	@DisplayName("Comprueba que hay victoria del atacante a mitad del tablero en una partida simple sin capturas.")
	void probarPartidaSimpleConVictoriaAtacanteCapturandoReyAMitadDeTablero() throws CoordenadasIncorrectasException {

		// given
		arbitro.mover(fabricarJugada(tablero, 3, 5, 2, 5)); // atacante
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 4, 3, 5)); // defensor
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 6, 3, 6, 4)); // atacante
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 5, 6, 5)); // defensor
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 5, 3, 5, 5)); // atacante
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 3, 3, 5)); // rey
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();

		// @formatter:off
		/* El estado del tablero antes del último movimiento debería ser:
		 * 7 - - - A - - -  
		 * 6 - - - A - - -
		 * 5 - - - D - A - 
		 * 4 A A D - - R A 
		 * 3 - - - D - - -
		 * 2 - - - - - A - 
		 * 1 - - - - A D -
		 *   a b c d e f g   
		 */
		 // @formatter:on

		// when
		arbitro.mover(fabricarJugada(tablero, 5, 5, 4, 5)); // atacante hace pinza sobre el rey en vertical
		// then
		final String cadenaEsperada = """
				7 - - - A - - -
				6 - - - A - - -
				5 - - - D - A -
				4 A A D - - R A
				3 - - - D - A -
				2 - - - - - - -
				1 - - - - A D -
				  a b c d e f g """.replaceAll("\\s", "");
		String cadenaObtenida = arbitro.consultarTablero().aTexto().replaceAll("\\s", "");
		assertAll("Victoria de atacante rodeando al rey a mitad de tablero.",
				() -> assertThat("Debería ganar el atacante.", arbitro.haGanadoAtacante(), is(true)),
				() -> assertThat("No debería ganar el defensor.", arbitro.haGanadoRey(), is(false)),
				() -> assertThat("Deberían estar todas las piezas atacantes sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
				() -> assertThat("Deberían estar todas las piezas defensoras sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
				() -> assertThat("Deberían estar todavía el rey sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)),
				() -> assertThat("La partida se desarrolla en un número erróneo de jugadas.",
						arbitro.consultarNumeroJugada(), is(7)),
				() -> assertThat("El turno debería ser del jugado atacante.", arbitro.consultarTurno(),
						is(Color.NEGRO)),
				() -> assertEquals(cadenaEsperada, cadenaObtenida, "Estado del tablero en formato texto incorrecto")
				);

	}
}
