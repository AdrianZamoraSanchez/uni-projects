package tafl.control.medio;

import static org.junit.jupiter.api.Timeout.ThreadMode.SEPARATE_THREAD;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Timeout;

import tafl.control.Arbitro;
import tafl.modelo.Tablero;

/**
 * Partida simple con pocos movimientos y finalización con victoria
 * de atacante o defensor, sin aplicar ninguna captura.
 * 
 * Demuestra la posibilidad de jugar una partida mínima sin incluir
 * la funcionalidad de capturas.
 *  
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@DisplayName("Tests del arbitro sobre partidas simples.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public abstract class ArbitroAbstractoPartidaSimpleTest {

	/** Árbitro de testing. */
	protected Arbitro arbitro;

	/** Tablero de testing. */
	protected Tablero tablero;
}
