package tafl.control.ardri.medio;

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

import tafl.control.ArbitroArdRi;
import tafl.excepcion.CoordenadasIncorrectasException;
import tafl.modelo.Tablero;
import tafl.util.Color;
import tafl.util.TipoPieza;

/**
 * Partida simple del ArbitroArdRi con pocos movimientos y finalización con victoria
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
@DisplayName("Tests del ArbitroArdRi sobre partidas simples.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public class ArbitroPartidaSimpleTest extends tafl.control.medio.ArbitroAbstractoPartidaSimpleTest {

	/** Generación del árbitro para testing. */
	@BeforeEach
	void inicializar() {
		// Inyección de tablero para testing...
		tablero = new Tablero();
		arbitro = new ArbitroArdRi(tablero);
		arbitro.colocarPiezasConfiguracionInicial();
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
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
		arbitro.mover(fabricarJugada(tablero, 3, 5, 2, 5)); // atacante f4f5
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 4, 4, 5, 4)); // defensor e3e2
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 2, 6, 1, 6)); // atacante g5g6
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 4, 4, 4)); // defensor e4e3
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();		
		arbitro.mover(fabricarJugada(tablero, 3, 6, 2, 6)); // atacante g4g5
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 3, 3, 4)); // rey d4e4
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 1, 4, 1)); // atacante b4b3
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 4, 3, 5)); // rey e4f4
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 4, 1, 5, 1)); // atacante b3b2
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();

		// @formatter:off
		/* El estado del tablero antes del último movimiento debería ser:
		 * 7 - - A A A - -  
		 * 6 - - - A - - A 
		 * 5 A - D D D A A 
		 * 4 A - D - - R - 
		 * 3 A - D D D - A
		 * 2 - A - A D - - 
		 * 1 - - A A A - -
		 *   a b c d e f g   
		 */
		 // @formatter:on

		// when
		arbitro.mover(fabricarJugada(tablero, 3, 5, 3, 6)); // rey alcanza lateral
		// then
		final String cadenaEsperada = """
				7 - - A A A - -  
				6 - - - A - - A 
				5 A - D D D A A 
				4 A - D - - - R 
				3 A - D D D - A
				2 - A - A D - - 
				1 - - A A A - -
				  a b c d e f g """.replaceAll("\\s", "");
		String cadenaObtenida = arbitro.consultarTablero().aTexto().replaceAll("\\s", "");
		assertAll("Victoria del rey alcanzando provincia en partida basica.",
				() -> assertThat("Debería ganar el defensor.", arbitro.haGanadoRey(), is(true)),
				() -> assertThat("No debería ganar el atacante.", arbitro.haGanadoAtacante(), is(false)),
				() -> assertThat("Deberían estar todas las piezas atacantes sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(16)),
				() -> assertThat("Deberían estar todas las piezas defensoras sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(8)),
				() -> assertThat("Deberían estar todavía el rey sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)),
				() -> assertThat("La partida se desarrolla en un número erróneo de jugadas.",
						arbitro.consultarNumeroJugada(), is(10)),
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
		arbitro.mover(fabricarJugada(tablero, 3, 5, 2, 5)); // atacante f4f5
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 4, 4, 5, 4)); // defensor e3e2
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 0, 4, 1, 4)); // atacante e7e6
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 4, 4, 4)); // defensor e4e3
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 0, 2, 1, 2)); // atacante c7c6
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 3, 3, 4)); // rey d4e4
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 4, 0, 5, 0)); // atacante a3a2
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 4, 3, 5)); // rey e4f4
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		
		// @formatter:off
		/* El estado del tablero antes del último movimiento debería ser:
		 * 7 - - - A - - -  
		 * 6 - - A A A - -
		 * 5 A - D D D A A 
		 * 4 A A D - - R A 
		 * 3 - - D D D - A
		 * 2 A - - A D- - - 
		 * 1 - - A A A - -
		 *   a b c d e f g   
		 */
		 // @formatter:on

		// when
		arbitro.mover(fabricarJugada(tablero, 4, 6, 4, 5)); // atacante hace pinza sobre el rey en vertical
		// then
		final String cadenaEsperada = """
				7 - - - A - - -
				6 - - A A A - -
				5 A - D D D A A
				4 A A D - - R A
				3 - - D D D A -
				2 A - - A D - -
				1 - - A A A - -
				  a b c d e f g """.replaceAll("\\s", "");
		String cadenaObtenida = arbitro.consultarTablero().aTexto().replaceAll("\\s", "");
		assertAll("Victoria de atacante rodeando al rey a mitad de tablero.",
				() -> assertThat("Debería ganar el atacante.", arbitro.haGanadoAtacante(), is(true)),
				() -> assertThat("No debería ganar el defensor.", arbitro.haGanadoRey(), is(false)),
				() -> assertThat("Deberían estar todas las piezas atacantes sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(16)),
				() -> assertThat("Deberían estar todas las piezas defensoras sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(8)),
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
