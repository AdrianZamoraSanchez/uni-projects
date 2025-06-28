package tafl.control.basico;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Timeout.ThreadMode.SEPARATE_THREAD;
import static tafl.control.TestUtil.fabricarJugada;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import tafl.control.Arbitro;
import tafl.modelo.Jugada;
import tafl.modelo.Tablero;

/**
 * Comprobación de movimientos ilegales.
 * 
 * Se asume que al menos se han colocado correctamente las piezas. En caso
 * contrario estos tests no se pasarán. Aplicable a ambos tipos de juego.
 * 
 * @see ArbitroAbstractoInicializacionTest
 */
@DisplayName("Tests del Arbitro sobre el control de movimientos ilegales.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public abstract class ArbitroAbstractoMovimientosIlegalesTest {

	/** Árbitro de testing. */
	protected Arbitro arbitro;

	/** Tablero de testing. */
	protected Tablero tablero;

	
	/**
	 * Comprobacion de ilegalidad de intentar mover desde un origen vacío.
	 * Aplicable a ambos tipos de juego.
	 * 
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento ilegal desde un origen vacío.")
	@Test
	void comprobarOrigenVacío() {
		// given
		// then
		Jugada jugada1 = fabricarJugada(tablero, 0, 0, 0, 1);
		Jugada jugada2 = fabricarJugada(tablero, 4, 1, 6, 1);
		Jugada jugada3 = fabricarJugada(tablero, 5, 5, 5, 6);
		Jugada jugada4 = fabricarJugada(tablero, 1, 6, 1, 5);
		assertAll("mover desde una celda vacía",
				() -> assertThat("El movimiento debería ser ilegal para " + jugada1, arbitro.esMovimientoLegal(jugada1),
						is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada2, arbitro.esMovimientoLegal(jugada2),
						is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada3, arbitro.esMovimientoLegal(jugada3),
						is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada4, arbitro.esMovimientoLegal(jugada4),
						is(false)));

	}
	
	/**
	 * Comprobacion de ilegalidad de intentar mover una pieza atacante a su misma coordenada.
	 * Aplicable a ambos tipos de juego.
	 * 
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento ilegal de una pieza atacante a su misma coordenada")
	@Test
	void comprobarMoverALaMismaCeldaAtacante() {
		// given
		// then
		Jugada jugada1 = fabricarJugada(tablero, 0, 3, 0, 3);
		Jugada jugada2 = fabricarJugada(tablero, 3, 0, 3, 0); 
		Jugada jugada3 = fabricarJugada(tablero, 3, 5, 3, 5); 
		Jugada jugada4 = fabricarJugada(tablero, 5, 3, 5, 3);		
		assertAll("mover atacante a su misma coordenada",
				() -> assertThat("El movimiento debería ser ilegal para " + jugada1, arbitro.esMovimientoLegal(jugada1), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada2, arbitro.esMovimientoLegal(jugada2), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada3, arbitro.esMovimientoLegal(jugada3), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada4, arbitro.esMovimientoLegal(jugada4), is(false))
				);
		
	}
}
