package tafl.control.brandubh.medio;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Timeout.ThreadMode.SEPARATE_THREAD;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import tafl.control.ArbitroBrandubh;
import tafl.modelo.Tablero;

/**
 * Validación de argumentos.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@DisplayName("Tests del ArbitroBrandubh sobre la validación de argumentos.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo
																			// los de ciclo de vida
public class ArbitroValidacionArgumentosTest extends tafl.control.medio.ArbitroAbstractoValidacionArgumentosTest {

	/** Generación del árbitro para testing. */
	@BeforeEach
	void inicializar() {
		// Inyección de tablero para testing...
		tablero = new Tablero();
		arbitro = new ArbitroBrandubh(tablero);
		arbitro.colocarPiezasConfiguracionInicial();
	}

	/**
	 * Comprueba que el constructor con tablero nulo provoca excepción.
	 * 
	 * @see tafl.control.ArbitroBrandubh
	 */
	@DisplayName("Comprueba que el constructor con tablero nulo lanza la excepción adecuada.")
	@Test
	void comprobarConstructorConTableroNulo() {
		// given
		Tablero tablero = null;
		// when
		// then
		assertThrows(IllegalArgumentException.class, () -> new ArbitroBrandubh(tablero));
	}
}
