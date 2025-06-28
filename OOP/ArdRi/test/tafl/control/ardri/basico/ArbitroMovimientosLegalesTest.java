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
import tafl.control.basico.ArbitroAbstractoInicializacionTest;
import tafl.excepcion.CoordenadasIncorrectasException;
import tafl.modelo.Jugada;
import tafl.modelo.Tablero;
import tafl.util.Color;
import tafl.util.TipoPieza;

/**
 * Comprobación del ArbitroArdRi de movimientos legales del árbitro.
 * 
 * Se asume que al menos se han colocado correctamente las piezas. En caso
 * contrario estos tests no se pasarán.
 * 
 * @see ArbitroAbstractoInicializacionTest
 */
@DisplayName("Tests del ArbitroArdRi sobre el control de  movimientos legales.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public class ArbitroMovimientosLegalesTest extends tafl.control.basico.ArbitroAbstractoMovimientosLegalesTest {

	/** 
	 * Generación del árbitro para testing.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 */
	// @formatter:off
	/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
     * 7 - - - A - - -  
	 * 6 - - - - - - - 
	 * 5 - - D - D - - 
	 * 4 A - - R - - A
	 * 3 - - D - D - -
	 * 2 - - - - - - - 
	 * 1 - - - A - - -
	 *   a b c d e f g   
	 */	
	 // @formatter:on
	@BeforeEach
	void inicializar() throws CoordenadasIncorrectasException {
		// Inyección de tablero para testing...
		tablero = new Tablero();
		arbitro = new ArbitroArdRi(tablero);
		arbitro.colocarPiezas(new TipoPieza[] {
				TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE,
				TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR,
				TipoPieza.REY}, 
				
				new int[][] {{0, 3}, {3, 0}, {3, 6}, {6, 3},
					// defensores
					{2, 2}, {2, 4}, {4, 2}, {4, 4},
					// rey
					{3, 3}
				},
				Color.NEGRO);
	}
	
	
	/**
	 * Comprobacion de legalidad de volver al mover el rey al trono tras abandonarlo.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento legal de volver a mover al rey al trono tras abandonarlo.")
	@Test
	void comprobarSePuedeVolverAMoverElReyAlTronoTrasAbandonarlo()  throws CoordenadasIncorrectasException {
		// given
		// when
		arbitro.cambiarTurno(); // cambiamos turno para que pueda mover legalmente el rey blanco
		arbitro.mover(fabricarJugada(tablero,3,3,3,4)); // mover rey y abandonar trono	
		//then
		Jugada jugada1 = fabricarJugada(tablero,3,4,3,3); // intentar mover rey otra vez al trono
		assertThat("El rey debería poder volver en su turno al trono para " + jugada1, arbitro.esMovimientoLegal(jugada1), is(true));
	}
	
	/**
	 * Comprobacion de legalidad de intentar mover atacante en el
	 * turno inicial.
	 * 
	 * @see tafl.control.TestUtil#fabricarJugada
	 */	
	@DisplayName("Comprueba el movimiento legal de mover atacante en el turno inicial.")
	@Test
	void comprobarLegalidadDeMoverPiezaAtacanteACeldaCorrecta() {
		Jugada jugada1 = fabricarJugada(tablero,0,3,0,2);
		Jugada jugada2 = fabricarJugada(tablero,3,0,4,0);
		Jugada jugada3 = fabricarJugada(tablero,6,3,6,4);
		Jugada jugada4 = fabricarJugada(tablero,3,6,2,6);
		assertAll("legalidad de mover atacante de forma correcta en turno inicial",
				() -> assertThat("El movimiento debería ser legal para " + jugada1, arbitro.esMovimientoLegal(jugada1), is(true)),
				() -> assertThat("El movimiento debería ser legal para " + jugada2, arbitro.esMovimientoLegal(jugada2), is(true)),
				() -> assertThat("El movimiento debería ser legal para " + jugada3, arbitro.esMovimientoLegal(jugada3), is(true)),
				() -> assertThat("El movimiento debería ser legal para " + jugada4, arbitro.esMovimientoLegal(jugada4), is(true))			
				);
	}
	
	/**
	 * Comprobacion de legalidad de intentar mover defensor en el
	 * turno inicial.
	 * 
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento legal de mover defensor en el turno inicial.")
	@Test
	void comprobarLegalidadDeMoverPiezaDefensorACeldaCorrecta() {
		// given
		Jugada jugada1 = fabricarJugada(tablero,2,2,2,1);
		Jugada jugada2 = fabricarJugada(tablero,4,2,5,2);
		Jugada jugada3 = fabricarJugada(tablero,4,4,4,5);
		Jugada jugada4 = fabricarJugada(tablero,2,4,2,3);
		// when
		arbitro.cambiarTurno(); // cambiamos turno para que mueva defensor
		// then
		assertAll("legalidad de mover defensor de forma correcta en turno inicial",
				() -> assertThat("El movimiento debería ser legal para " + jugada1, arbitro.esMovimientoLegal(jugada1), is(true)),
				() -> assertThat("El movimiento debería ser legal para " + jugada2, arbitro.esMovimientoLegal(jugada2), is(true)),
				() -> assertThat("El movimiento debería ser legal para " + jugada3, arbitro.esMovimientoLegal(jugada3), is(true)),
				() -> assertThat("El movimiento debería ser legal para " + jugada4, arbitro.esMovimientoLegal(jugada4), is(true))			
				);
	}
}
