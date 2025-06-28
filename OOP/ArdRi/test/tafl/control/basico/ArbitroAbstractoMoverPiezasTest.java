package tafl.control.basico;

import static org.junit.jupiter.api.Timeout.ThreadMode.SEPARATE_THREAD;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Timeout;

import tafl.control.Arbitro;
import tafl.modelo.Tablero;

/**
 * Comprobación de los movimientos de piezas sobre el tablero
 * trasladando las piezas. 
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 */
@DisplayName("Tests del Arbitro sobre el movimiento de piezas.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public abstract class ArbitroAbstractoMoverPiezasTest {
	
	/** Árbitro de testing. */
	protected Arbitro arbitro;

	/** Tablero de testing. */
	protected Tablero tablero;
}
