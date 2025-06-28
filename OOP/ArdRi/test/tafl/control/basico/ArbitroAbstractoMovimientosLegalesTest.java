package tafl.control.basico;

import static org.junit.jupiter.api.Timeout.ThreadMode.SEPARATE_THREAD;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Timeout;

import tafl.control.Arbitro;
import tafl.modelo.Tablero;

/**
 * Comprobación de movimientos legales del árbitro.
 * 
 * Se asume que al menos se han colocado correctamente las piezas. En caso
 * contrario estos tests no se pasarán. 
 * 
 * @see ArbitroAbstractoInicializacionTest
 */
@DisplayName("Tests del Arbitro sobre el control de  movimientos legales.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public abstract class ArbitroAbstractoMovimientosLegalesTest {

	
	/** Árbitro de testing. */
	protected Arbitro arbitro;

	/** Tablero de testing. */
	protected Tablero tablero;
}
