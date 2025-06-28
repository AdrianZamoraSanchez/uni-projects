package brandubh.util;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertAll;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * Tests sobre coordenada.
 * 
 * Dado que se genera con un tipo record la realización es casi redundante.
 * 
 * @author <a href="mailto:rmartico@ubu.es">Raúl Marticorena Sánchez</a>
 * @version 1.0 20220803
 * 
 */
@DisplayName("Tests sobre coordenada (sin dependencias de otras clases).")
@Timeout(value = 1, unit = TimeUnit.SECONDS) // Time out global para todos los tests salvo los de ciclo de vida
public class CoordenadaTest {
	
	/** Coordenada. */
	private Coordenada coordenada00;
	/** Coordenada. */
	private Coordenada coordenada23;
	/** Coordenada. */
	private Coordenada coordenada66;
	
	/**
	 * Inicialización.
	 */
	@BeforeEach
	void inicializar() {
		coordenada00 = new Coordenada(0,0);
		coordenada23 = new Coordenada(2,3);
		coordenada66 = new Coordenada(6,6);
	}
	
	/**
	 * Comprueba la correcta inicialización de fila.
	 */
	@DisplayName("Comprueba la inicialización de fila.")
	@Test
	public void probarInicializacionDeFila() {
		assertAll("comprobando la correcta inicialización de fila ",
				() -> assertThat("El valor de la fila para la coordenada es incorrecto.", 
				coordenada00.fila(), is(0)),
				() -> assertThat("El valor de la fila para la coordenada es incorrecto.", 
				coordenada23.fila(), is(2)),
				() -> assertThat("El valor de la fila para la coordenada es incorrecto.", 
				coordenada66.fila(), is(6)));
	}
	
	/**
	 * Comprueba la correcta inicialización de columna.
	 */
	@DisplayName("Comprueba la inicialización de columna.")
	@Test
	public void probarInicializacionDeColumna() {
		assertAll("comprobando la correcta inicialización de columna ",
				() -> assertThat("El valor de la columna para la coordenada es incorrecto.", 
				coordenada00.columna(), is(0)),
				() -> assertThat("El valor de la columna para la coordenada es incorrecto.", 
				coordenada23.columna(), is(3)),
				() -> assertThat("El valor de la columna para la coordenada es incorrecto.", 
				coordenada66.columna(), is(6)));
	}
}
