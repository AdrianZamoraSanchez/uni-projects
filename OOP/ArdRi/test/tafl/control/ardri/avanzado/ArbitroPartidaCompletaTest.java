package tafl.control.ardri.avanzado;

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
 * Partida completa con ArbitroArdRi y finalización con victoria
 * de atacante o defensor, aplicando además alguna captura.
 * 
 * Demuestra la posibilidad de jugar una partida completa 
 * incluyendo además la captura de piezas.
 *  
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@DisplayName("Tests del ArbitroArdRi sobre victorias avanzadas completas.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public class ArbitroPartidaCompletaTest extends tafl.control.avanzado.ArbitroAbstractoPartidaCompletaTest {

	/** Generación del árbitro para testing. */
	@BeforeEach
	void inicializar() {
		// Inyección de tablero para testing...
		tablero = new Tablero();
		arbitro = new ArbitroArdRi(tablero);
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
		arbitro.mover(fabricarJugada(tablero, 3, 5, 2, 5)); // atacante f4f5
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 4, 4, 5, 4)); // defensor e3e2
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 2, 5, 1, 5)); // atacante f5f6
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 4, 4, 4)); // defensor e4e3
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 0, 4, 1, 4)); // atacante e7e6
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 3, 3, 4)); // rey d4e4
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 4, 6, 5, 6)); // atacante g3g2
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 5, 4, 5, 5)); // defensor e2f2
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 6, 4, 5, 4)); // atacante e1e2 se captura pieza blanca en f2
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 4, 3, 5)); // rey e4f4
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 6, 4, 6)); // atacante g4g3
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();

		// when
		arbitro.mover(fabricarJugada(tablero, 3, 5, 3, 6)); // f4g4 rey llega a borde
		// el rey no se capura y no se cambia el turno
		// para observar el estado de la partida en el test

		// @formatter:off
		/* El estado del tablero debería ser:
		 * 7 - - A A - - -  
		 * 6 - - - A A A -
		 * 5 A - D D D - A 
		 * 4 A A D - - - R 
		 * 3 A - D D D - A
		 * 2 - - - A A - A 
		 * 1 - - A A - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on

		// then
		final String cadenaEsperada = """
				7 - - A A - - -
				6 - - - A A A -
				5 A - D D D - A
				4 A A D - - - R
				3 A - D D D - A
				2 - - - A A - A
				1 - - A A - - -
				  a b c d e f g """.replaceAll("\\s", "");
		String cadenaObtenida = arbitro.consultarTablero().aTexto().replaceAll("\\s", "");
		
		assertAll("Victoria del defensor llegando al borde.",
				
				() -> assertThat("No debería ganar el atacante.", arbitro.haGanadoAtacante(), is(false)),
				() -> assertThat("Debería ganar el defensor.", arbitro.haGanadoRey(), is(true)),
				() -> assertThat("Deberían estar todas las piezas atacantes sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(16)),
				() -> assertThat("Deberían estar solo siete defensores sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(7)), // se ha realizado una captura
				() -> assertThat("Deberían estar todavía el rey sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)),
				() -> assertThat("La partida se desarrolla en un número erróneo de jugadas.",
						arbitro.consultarNumeroJugada(), is(12)),
				() -> assertThat("El turno debería ser del jugado defensor.", arbitro.consultarTurno(),
						is(Color.BLANCO)),
				() -> assertEquals(cadenaEsperada, cadenaObtenida, "Estado del tablero en formato texto incorrecto")
				);
	}

	/**
	 * Comprueba una partida con victoria del atacante con una doble captura de piezas del defensor
	 * y captura simple de atancante.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 */
	@Test
	@DisplayName("Comprueba que hay victoria del atacante con una doble captura de piezas del defensor y captura simple del atacante.")
	void probarPartidaPartidaConVictoriaAtacanteCapturandoReyAMitadDeTableroConDobleCapturaDePiezasDelDefensorYCapturaSimpleDelAtacante()  throws CoordenadasIncorrectasException {

		// given
		arbitro.mover(fabricarJugada(tablero, 4, 6, 5, 6)); // atacante g3g2
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 4, 4, 4, 5)); // defensor e3f3
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 2, 6, 2, 5)); // atacante g5f5
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 4, 4, 4)); // defensor e4e3
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 5, 3, 4)); // atacante f4e4
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 4, 5, 5, 4)); // defensor f3f2
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 6, 4, 5, 4)); // atacante e1e2 realiza doble captura de defensor
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 4, 3, 4, 4)); // defensor d3e3 captura simple de atacante
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();		
		arbitro.mover(fabricarJugada(tablero, 3, 6, 4, 6)); // atacante g4g3
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 3, 3, 4)); // rey d4e4
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 5, 6, 5, 5)); // atacante g2f2
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		arbitro.mover(fabricarJugada(tablero, 3, 4, 3, 5)); // rey e4f4
		arbitro.realizarCapturasTrasMover();
		arbitro.cambiarTurno();
		// when
		arbitro.mover(fabricarJugada(tablero, 4, 6, 4, 5)); // atacante g3f3 captura el rey
		// no se realizan capturas ni se cambia el turno
		// para observar el estado de la partida en el test

		// @formatter:off
		/* El estado del tablero debería ser:
		 * 7 - - A A A - -  
		 * 6 - - - A - - -
		 * 5 A - D D D A - 
		 * 4 A A D - - R - 
		 * 3 A - D - D A -
		 * 2 - - - A A A - 
		 * 1 - - A A - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on

		// then
		final String cadenaEsperada = """
				7 - - A A A - -
				6 - - - A - - -
				5 A - D D D A -
				4 A A D - - R -
				3 A - D - D A -
				2 - - - A A A -
				1 - - A A - - -
				  a b c d e f g """.replaceAll("\\s", "");
		String cadenaObtenida = arbitro.consultarTablero().aTexto().replaceAll("\\s", "");
		
		assertAll("Victoria de atacante rodeando al rey con dos atacantes.",
				
				() -> assertThat("Debería ganar el atacante.", arbitro.haGanadoAtacante(), is(true)),
				() -> assertThat("No debería ganar el defensor.", arbitro.haGanadoRey(), is(false)),
				() -> assertThat("No deberían estar todas las piezas atacantes sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(15)), // se captura un atacante
				() -> assertThat("Deberían estar solo seis defensores sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(6)), // se capturan dos defensores
				() -> assertThat("Deberían estar todavía el rey sobre el tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)),
				() -> assertThat("La partida se desarrolla en un número erróneo de jugadas.",
						arbitro.consultarNumeroJugada(), is(13)),
				() -> assertThat("El turno debería ser del jugado atacante.", arbitro.consultarTurno(),
						is(Color.NEGRO)),
				() -> assertEquals(cadenaEsperada, cadenaObtenida, "Estado del tablero en formato texto incorrecto")
				);
	}
}
