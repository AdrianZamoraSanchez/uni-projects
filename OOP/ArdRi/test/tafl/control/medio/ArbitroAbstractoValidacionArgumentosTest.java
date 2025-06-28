package tafl.control.medio;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Timeout.ThreadMode.SEPARATE_THREAD;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import tafl.control.Arbitro;
import tafl.excepcion.CoordenadasIncorrectasException;
import tafl.modelo.Celda;
import tafl.modelo.Jugada;
import tafl.modelo.Tablero;
import tafl.util.Color;
import tafl.util.Coordenada;
import tafl.util.TipoPieza;

/**
 * Validación de argumentos en métodos que reciben valores incorrectos
 * y se provoca el lanzamiento de excepción. Aplicable a ambos tipos de juego.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@DisplayName("Tests del arbitro sobre argumentos inválidos en los métodos.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo
																			// los de ciclo de vida
public abstract class ArbitroAbstractoValidacionArgumentosTest {

	/** Árbitro de testing. */
	protected Arbitro arbitro;

	/** Tablero de testing. */
	protected Tablero tablero;

	/**
	 * Comprueba que la consulta de movimiento legal con jugada nula provoca excepción.
	 * 
	 * @see tafl.control.Arbitro#esMovimientoLegal(Jugada)
	 */
	@DisplayName("Comprueba que la consulta de movimiento legal con jugada nula lanza la excepción adecuada.")
	@Test
	void comprobarEsMovimientoLegalConJugadaNula() {
		// given
		Jugada jugada = null;
		// when
		// then
		assertThrows(IllegalArgumentException.class, () -> arbitro.esMovimientoLegal(jugada));
	}

	/**
	 * Comprueba que la colocación de piezas con tipos de piezas con valor nulo
	 * provoca excepción.
	 * 
	 * @see tafl.control.Arbitro#colocarPiezas(TipoPieza[], int[][], Color)
	 */
	@DisplayName("Comprueba que la colocación de tipos de piezas con valor nulo lanza la excepción adecuada.")
	@Test
	void comprobarColocarPiezasConTiposDePiezaNulo() {
		// given
		// when
		// then
		assertThrows(IllegalArgumentException.class, () -> arbitro.colocarPiezas(null, new int[][] {}, Color.BLANCO));
	}

	/**
	 * Comprueba que la colocación de piezas con array de coordenadas nulo provoca
	 * excepción.
	 * 
	 * @see tafl.control.Arbitro#colocarPiezas(TipoPieza[], int[][], Color)
	 */
	@DisplayName("Comprueba que la colocación de piezas con array de coordenadas nulo lanza la excepción adecuada.")
	@Test
	void comprobarColocarPiezasConArrayCoordenadasNulo() {
		// given
		// when
		// then
		assertThrows(IllegalArgumentException.class,
				() -> arbitro.colocarPiezas(new TipoPieza[] {}, null, Color.NEGRO));
	}

	/**
	 * Comprueba que la colocación de piezas de color con valor nulo provoca
	 * excepción.
	 * 
	 * @see tafl.control.Arbitro#colocarPiezas(TipoPieza[], int[][], Color)
	 */
	@DisplayName("Comprueba que la colocación de color con valor nulo lanza la excepción adecuada.")
	@Test
	void comprobarColocarPiezasConColorNulo() {
		// given
		// when
		// then
		assertThrows(IllegalArgumentException.class,
				() -> arbitro.colocarPiezas(new TipoPieza[] {}, new int[][] {}, null));
	}

	/**
	 * Comprueba que la colocación con coordenada incorrecta lanza excepción
	 * provoca excepción.
	 * 
	 * @see tafl.control.Arbitro#colocarPiezas(TipoPieza[], int[][], Color)
	 */
	@DisplayName("Comprueba que la colocación con coordenada incorrecta lanza la excepción adecuada.")
	@Test
	void comprobarColocarPiezasConCoordenadasIncorrectas() {
		// given
		// when
		// then
		assertThrows(CoordenadasIncorrectasException.class,
				() -> arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE }, new int[][] { {-1, -1} }, Color.BLANCO));
	}
	
	/**
	 * Comprueba que mover con jugada nula provoca excepción.
	 * 
	 * @see tafl.control.Arbitro#mover(Jugada)
	 */
	@DisplayName("Comprueba que mover con jugada  nula lanza la excepción adecuada.")
	@Test
	void comprobarMoverConJugadaNula() {
		// given
		Jugada jugada = null;
		// when
		// then
		assertThrows(IllegalArgumentException.class, () -> arbitro.mover(jugada));
	}
	
	/**
	 * Comprueba que mover con jugada con coordenada origen incorrecta provoca excepción.
	 * 
	 * @see tafl.control.Arbitro#mover(Jugada)
	 * @version 1.0.0.1
	 */
	@DisplayName("Comprueba que mover con jugada con coordenada origen incorrecta lanza la excepción adecuada.")
	@Test
	void comprobarMoverConCoordenadaOrigenIncorrectas() {
		// given
		Celda origenIncorrecto = new Celda(new Coordenada(-1,-1));
		Celda destinoCorrecto = new Celda(new Coordenada(0,0));
		Jugada jugadaIncorrecta = new Jugada(origenIncorrecto, destinoCorrecto);
		// when
		// then
		assertThrows(CoordenadasIncorrectasException.class, () -> arbitro.mover(jugadaIncorrecta)); // Changed
	}
	
	/**
	 * Comprueba que mover con jugada con coordenada destino incorrecta provoca excepción.
	 * 
	 * @see tafl.control.Arbitro#mover(Jugada)
	 * @version 1.0.0.1
	 */
	@DisplayName("Comprueba que mover con jugada con coordenada destino incorrecta lanza la excepción adecuada.")
	@Test
	void comprobarMoverConCoordenadaDestinoIncorrectas() {
		// given
		Celda origenCorrecto = new Celda(new Coordenada(0,0));
		Celda destinoIncorrecto = new Celda(new Coordenada(7,7));
		Jugada jugadaIncorrecta = new Jugada(origenCorrecto, destinoIncorrecto);
		// when
		// then
		assertThrows(CoordenadasIncorrectasException.class, () -> arbitro.mover(jugadaIncorrecta)); // Changed
	}
	
	/**
	 * Comprueba que mover con jugada con ambas coordenadas incorrectas provoca excepción.
	 * 
	 * @see tafl.control.Arbitro#mover(Jugada)
	 * @version 1.0.0.1
	 */
	@DisplayName("Comprueba que mover con jugada con ambas coordenadas incorrectas lanza la excepción adecuada.")
	@Test
	void comprobarMoverConAmbasCoordenadasIncorrectas() {
		// given
		Celda origenIncorrecto = new Celda(new Coordenada(-1,0));
		Celda destinoIncorrecto = new Celda(new Coordenada(0,7));
		Jugada jugadaIncorrecta = new Jugada(origenIncorrecto, destinoIncorrecto);
		// when
		// then
		assertThrows(CoordenadasIncorrectasException.class, () -> arbitro.mover(jugadaIncorrecta)); // Changed
	}
	
	
}
