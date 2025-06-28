package tafl.control;

import static org.junit.jupiter.params.provider.Arguments.arguments;

import java.util.stream.Stream;

import org.junit.jupiter.params.provider.Arguments;

import tafl.excepcion.CoordenadasIncorrectasException;
import tafl.modelo.Jugada;
import tafl.modelo.Pieza;
import tafl.modelo.Tablero;
import tafl.util.Coordenada;
import tafl.util.TipoPieza;

/**
 * Clase de utilidad para generar jugadas en los tests.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 */
public class TestUtil {

	/**
	 * Fabrica una instancia de jugada a partir de las coordenadas de origen y
	 * destino.
	 * 
	 * @param tablero        tablero
	 * @param filaOrigen     fila origen
	 * @param columnaOrigen  columna origen
	 * @param filaDestino    fila destino
	 * @param columnaDestino columna destino
	 * @return jugada
	 * @throws IllegalArgumentException si el tablero es nulo
	 * @throws RuntimeException si alguno de los argumentos es incorrecto
	 */
	public static Jugada fabricarJugada(Tablero tablero, int filaOrigen, int columnaOrigen, int filaDestino,
			int columnaDestino) throws IllegalArgumentException, RuntimeException {
		if (tablero == null) throw new IllegalArgumentException("El tablero no puede ser nulo.");
		try {
			return new Jugada(tablero.consultarCelda(new Coordenada(filaOrigen, columnaOrigen)),
					tablero.consultarCelda(new Coordenada(filaDestino, columnaDestino)));

		} catch (CoordenadasIncorrectasException ex) {
			throw new RuntimeException("Error con coordenadas al generar una jugada válida para testeo.", ex);
		}
	}

	/**
	 * Proveedor de piezas atacantes del Brandubh.
	 * 
	 * @return pieza y coordenada de atacantes
	 */
	static Stream<Arguments> piezaYCoordenadaDeAtacanteBrandubhProvider() {
		return Stream.of(arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(0, 3)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(1, 3)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(3, 0)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(3, 1)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(3, 5)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(3, 6)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(5, 3)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(6, 3)));
	}

	/**
	 * Proveedor de piezas defensoras, sin incluir al rey del Brandubh.
	 * 
	 * @return pieza y coordenada de defensores
	 */
	static Stream<Arguments> piezaYCoordenadaDeDefensorBrandubhProvider() {
		return Stream.of(arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(2, 3)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(4, 3)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(3, 2)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(3, 4)));
	}

	/**
	 * Proveedor de piezas atacantes del ArdRi.
	 * 
	 * @return pieza y coordenada de atacantes
	 */
	static Stream<Arguments> piezaYCoordenadaDeAtacanteArdRiProvider() {
		return Stream.of(arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(0, 2)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(0, 3)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(0, 4)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(1, 3)),

				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(2, 0)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(3, 0)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(4, 0)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(3, 1)),

				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(3, 5)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(2, 6)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(3, 6)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(4, 6)),

				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(5, 3)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(6, 2)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(6, 3)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(6, 4)));
	}

	/**
	 * Proveedor de piezas defensoras, sin incluir al rey del ArdRi.
	 * 
	 * @return pieza y coordenada de defensores
	 */
	static Stream<Arguments> piezaYCoordenadaDeDefensorArdRiProvider() {
		return Stream.of(arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(2, 2)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(2, 3)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(2, 4)),

				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(3, 2)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(3, 4)),

				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(4, 2)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(4, 3)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(4, 4)));
	}

}
