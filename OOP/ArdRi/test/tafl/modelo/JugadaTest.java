package tafl.modelo;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Timeout.ThreadMode.SEPARATE_THREAD;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import tafl.util.Coordenada;
import tafl.util.Sentido;
import tafl.util.TipoCelda;

/**
 * Tests sobre Jugada.
 * 
 * Al ser un tipo record se minimizan los tests.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@DisplayName("Tests sobre la jugada.")
@Timeout(value = 1, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public class JugadaTest {
	
	/**
	 * Comprueba la deteccion de movimiento en horizontal.
	 */
	@DisplayName("Comprobar la detección de movimiento en horizontal.")
	@Test
	public void comprobarDeteccionDeMovimientoHorizontal() {
		Celda origen = new Celda(new Coordenada(0,0), TipoCelda.NORMAL);
		Celda destino = new Celda(new Coordenada(0,6), TipoCelda.NORMAL);
		Jugada jugada = new Jugada(origen, destino);
		assertThat("No identifica bien un movimiento horizontal.", jugada.esMovimientoHorizontalOVertical(), is(true));
	}
	
	/**
	 * Comprueba la deteccion de movimiento en vertical.
	 */
	@DisplayName("Comprobar la detección de movimiento en horizontal.")
	@Test
	public void comprobarDeteccionDeMovimientoVertical() {
		Celda origen = new Celda(new Coordenada(0,0), TipoCelda.NORMAL);
		Celda destino = new Celda(new Coordenada(6,0), TipoCelda.NORMAL);
		Jugada jugada = new Jugada(origen, destino);
		assertThat("No identifica bien un movimiento horizontal.", jugada.esMovimientoHorizontalOVertical(), is(true));
	}
	
	/**
	 * Comprueba la deteccion de movimiento en diagonal.
	 */
	@DisplayName("Comprobar la detección de movimiento en horizontal.")
	@Test
	public void comprobarDeteccionDeMovimientoDiagonal() {
		Celda origen = new Celda(new Coordenada(0,0), TipoCelda.NORMAL);
		Celda destino = new Celda(new Coordenada(6,6), TipoCelda.NORMAL);
		Jugada jugada = new Jugada(origen, destino);
		assertThat("No identifica bien un movimiento diagonal.", jugada.esMovimientoHorizontalOVertical(), is(false));
	}
	
	/**
	 * Comprueba la deteccion de movimiento con salto tipo caballo de ajedrez.
	 */
	@DisplayName("Comprobar la detección de movimiento con salto tipo caballo de ajedrez.")
	@Test
	public void comprobarDeteccionDeMovimientoConSalto() {
		Celda origen = new Celda(new Coordenada(0,0), TipoCelda.NORMAL);
		Celda destino = new Celda(new Coordenada(1,3), TipoCelda.NORMAL);
		Jugada jugada = new Jugada(origen, destino);
		assertThat("No identifica bien un movimiento con salto de caballo.", jugada.esMovimientoHorizontalOVertical(), is(false));
	}
	
	/**
	 * Comprueba la detección del sentido de movimiento entre dos celdas origen y
	 * destino que estén en horizontal o vertical.
	 */
	@DisplayName("Detectar el sentido en el movimiento horizontal o vertical entre dos celdas")
	@Test
	void comprobarSentidoEnMovimientoHorizonalOVertical() {
		assertAll(
				() -> assertThat("Mal detectado el sentido",
						new Jugada(new Celda(new Coordenada(0, 0), TipoCelda.NORMAL),
								new Celda(new Coordenada(1, 0), TipoCelda.NORMAL)).consultarSentido(),
						is(Sentido.VERTICAL_S)),
				() -> assertThat("Mal detectado el sentido",
						new Jugada(new Celda(new Coordenada(7, 7), TipoCelda.NORMAL),
								new Celda(new Coordenada(6, 7), TipoCelda.NORMAL)).consultarSentido(),
						is(Sentido.VERTICAL_N)),
				() -> assertThat("Mal detectado el sentido",
						new Jugada(new Celda(new Coordenada(2, 2), TipoCelda.NORMAL),
								new Celda(new Coordenada(2, 3), TipoCelda.NORMAL)).consultarSentido(),
						is(Sentido.HORIZONTAL_E)),
				() -> assertThat("Mal detectado el sentido",
						new Jugada(new Celda(new Coordenada(5, 5), TipoCelda.NORMAL),
								new Celda(new Coordenada(5, 4), TipoCelda.NORMAL)).consultarSentido(),
						is(Sentido.HORIZONTAL_O)));
	}
}
