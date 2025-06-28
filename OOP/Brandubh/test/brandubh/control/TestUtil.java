package brandubh.control;

import static org.junit.jupiter.params.provider.Arguments.arguments;

import java.util.stream.Stream;

import org.junit.jupiter.params.provider.Arguments;

import brandubh.modelo.Jugada;
import brandubh.modelo.Pieza;
import brandubh.modelo.Tablero;
import brandubh.util.Coordenada;
import brandubh.util.TipoPieza;

/**
 * Clase de utilidad para generar jugadas en los tests.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 */
public class TestUtil {
	
	/**
	 * Fabrica una instancia de jugada a partir de las coordenadas de origen y destino.
	 * 
	 * @param tablero tablero
	 * @param filaOrigen fila origen
	 * @param columnaOrigen columna origen
	 * @param filaDestino fila destino
	 * @param columnaDestino columan destino
	 * @return jugada
	 */
	public static Jugada fabricarJugada(Tablero tablero, int filaOrigen, int columnaOrigen, int filaDestino, int columnaDestino) {
		return new  Jugada(
				tablero.consultarCelda(	new Coordenada(filaOrigen, columnaOrigen)), 
				tablero.consultarCelda(new Coordenada(filaDestino, columnaDestino)));
	}

	/**
	 * Proveedor de piezas atacantes.
	 * 
	 * @return pieza y coordenada de atacantes
	 */
	static Stream<Arguments> piezaYCoordenadaDeAtacanteProvider() {
		return Stream.of(
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(0,3)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(1,3)), 
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(3,0)), 
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(3,1)), 
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(3,5)), 
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(3,6)), 
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(5,3)), 
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(6,3)));				
	}

	/**
	 * Proveedor de piezas defensoras, sin incluir al rey.
	 * 
	 * @return pieza y coordenada de defensores
	 */
	static Stream<Arguments> piezaYCoordenadaDeDefensorProvider() {
		return Stream.of(
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(2,3)),
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(4,3)), 
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(3,2)), 
				arguments(new Pieza(TipoPieza.ATACANTE), new Coordenada(3,4)));
	}	
	
}
