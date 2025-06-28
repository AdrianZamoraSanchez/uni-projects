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

/**
 * Comprobación de movimientos legales del árbitro.
 * 
 * Se asume que al menos se han colocado correctamente las piezas. En caso
 * contrario estos tests no se pasarán.
 * 
 * @see ArbitroInicializacionTest
 */
@DisplayName("Tests del Arbitro sobre el control de  movimientos legales.")
@Timeout(value = 2, unit = TimeUnit.SECONDS) // Time out global para todos los tests salvo los de ciclo de vida
public class ArbitroMovimientosLegalesTest {

	
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
	 * Comprobacion de legalidad de intentar mover atacante en el
	 * turno inicial.
	 * 
	 * @see brandubh.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento legal de mover atacante en el turno inicial.")
	@Test
	void comprobarLegalidadDeMoverPiezaAtacanteACeldaCorrecta() {
		Jugada jugada1 = fabricarJugada(tablero,0,3,0,2);
		Jugada jugada2 = fabricarJugada(tablero,3,0,4,0);
		Jugada jugada3 = fabricarJugada(tablero,5,3,5,4);
		Jugada jugada4 = fabricarJugada(tablero,3,6,1,6);
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
	 * @see brandubh.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento legal de mover defensor en el turno inicial.")
	@Test
	void comprobarLegalidadDeMoverPiezaDefensorACeldaCorrecta() {
		// given
		Jugada jugada1 = fabricarJugada(tablero,2,3,2,6);
		Jugada jugada2 = fabricarJugada(tablero,3,2,0,2);
		Jugada jugada3 = fabricarJugada(tablero,3,4,6,4);
		Jugada jugada4 = fabricarJugada(tablero,4,3,4,0);
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
	
	/**
	 * Comprobacion de legalidad de desplazar pieza defensora sobre el trono pero sin parar sobre dicha celda.
	 * 
	 * @see brandubh.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento legal de pieza defensora sobre el trono una vez abandonado sin parar en dicha celda.")
	@Test
	void comprobarSePuedeMoverSobreLaCeldaTronoUnDefensorSinPararSobreElla() {
		// given
		// when
		arbitro.mover(fabricarJugada(tablero,3,4,6,4)); // defensor
		arbitro.mover(fabricarJugada(tablero,3,3,3,4)); // mover rey y abandonar trono		
		arbitro.mover(fabricarJugada(tablero,4,3,4,6)); // defensor
		arbitro.cambiarTurno(); // mueve el defensor
		//then		
		Jugada jugada1 = fabricarJugada(tablero,2,3,4,3); // intentar mover defensor a una celda pasando por el trono
		assertThat("El movimiento debería ser legal para " + jugada1, arbitro.esMovimientoLegal(jugada1), is(true));
	}
	
	/**
	 * Comprobacion de legalidad de desplazar pieza atancante sobre el trono pero sin parar sobre dicha celda.
	 * 
	 * @see brandubh.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento legal de pieza atacante sobre el trono una vez abandonado sin parar en dicha celda.")
	@Test
	void comprobarSePuedeMoverSobreLaCeldaTronoUnAtacanteSinPararSobreElla() {
		// given
		// when
		arbitro.mover(fabricarJugada(tablero,3,4,6,4)); // defensor
		arbitro.mover(fabricarJugada(tablero,3,3,3,4)); // mover rey y abandonar trono		
		arbitro.mover(fabricarJugada(tablero,4,3,4,6)); // defensor
		arbitro.mover(fabricarJugada(tablero,2,3,2,6)); // defensor
		//then
		Jugada jugada1 = fabricarJugada(tablero,1,3,4,3); // intentar mover atacante a una celda pasando por el trono
		assertThat("El movimiento debería ser legal para " + jugada1, arbitro.esMovimientoLegal(jugada1), is(true));
	}
	
	
	/**
	 * Comprobacion de legalidad de volver al mover el rey al trono tras abandonarlo.
	 * 
	 * @see brandubh.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento legal de volver a mover al rey al trono tras abandonarlo.")
	@Test
	void comprobarSePuedeVolverAMoverElReyAlTronoTrasAbandonarlo() {
		// given
		// when
		arbitro.mover(fabricarJugada(tablero,3,4,6,4)); // defensor
		arbitro.mover(fabricarJugada(tablero,3,3,3,4)); // mover rey y abandonar trono	
		arbitro.cambiarTurno(); // cambiamos turno para que pueda mover legalmente el rey blanco
		//then
		Jugada jugada1 = fabricarJugada(tablero,3,4,3,3); // intentar mover rey otra vez al trono
		assertThat("El rey debería poder volver en su turno al trono para " + jugada1, arbitro.esMovimientoLegal(jugada1), is(true));
	}
}
