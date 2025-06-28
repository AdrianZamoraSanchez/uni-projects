package tafl.control.brandubh.medio;

import static org.junit.jupiter.api.Timeout.ThreadMode.SEPARATE_THREAD;
import static tafl.control.TestUtil.fabricarJugada;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import tafl.control.ArbitroBrandubh;
import tafl.excepcion.CoordenadasIncorrectasException;
import tafl.modelo.Tablero;
import tafl.util.Color;
import tafl.util.TipoPieza;

/**
 * Comprobación con ArbitroBrandubh de victoria de atacante rodeando al rey.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@DisplayName("Tests del ArbitroBrandubh sobre victorias del atacante rodeando al rey.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public class ArbitroVictoriaAtacanteTest extends tafl.control.medio.ArbitroAbstractoVictoriaAtacanteTest {

	/** Generación del árbitro para testing. */
	@BeforeEach
	void inicializar() {
		// Inyección de tablero para testing...
		tablero = new Tablero();
		arbitro = new ArbitroBrandubh(tablero);
	}
	
	/**
	 * Victoria de atacantes rodeando al rey con dos atacantes sin estar situado el
	 * trono.
	 */
	@Nested
	@DisplayName("Tests para comprobar que hay victoria del atacante si rodea con dos atacantes al rey cuando no está en el trono ni contiguo a él.")
	class VictoriasRodeandoAlReyPorDosLadosConAtacantesSinEstarEnElTronoOContiguoAlTrono {

		/**
		 * Comprueba victoria rodeando al rey con dos piezas atacantes en el borde
		 * superior.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarVictoriaRodeandoAlReyPorDosLadosEnHorizontalEnBordeSuperior()  throws CoordenadasIncorrectasException {
			rodearAlReyConDosAtacantesEnHorizontalEnBordeSuperior();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con dos piezas atacantes en el borde
		 * inferior.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarVictoriaRodeandoAlReyPorDosLadosEnHorizontalEnBordeInferior()  throws CoordenadasIncorrectasException {
			rodearAlReyConDosAtacantesEnHorizontalEnBordeInferior();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con dos piezas atacantes en el borde
		 * izquierdo.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarVictoriaRodeandoAlReyPorDosLadosEnHorizontalEnBordeIzquierdo()  throws CoordenadasIncorrectasException {
			rodearAlReyConDosAtacantesEnHorizontalEnBordeIzquierdo();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con dos piezas atacantes en el borde
		 * derecho.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarVictoriaRodeandoAlReyPorDosLadosEnHorizontalEnBordeDerecho()  throws CoordenadasIncorrectasException {
			rodearAlReyConDosAtacantesEnHorizontalEnBordeDerecho();
			comprobarReyRodeado();
		}

		/**
		 * Coloca piezas rodeando al rey de dos atacantes en horizontal en borde
		 * superior.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - A - R A - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - -
		 * 4 - - - - - - -
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConDosAtacantesEnHorizontalEnBordeSuperior()  throws CoordenadasIncorrectasException {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 0, 1 }, { 0, 4 }, { 0, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 0, 1, 0, 2));
		}

		/**
		 * Coloca piezas rodeando al rey de dos atacantes en horizontal en borde
		 * inferior.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - -
		 * 4 - - - - - - -
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - A R - A -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConDosAtacantesEnHorizontalEnBordeInferior()  throws CoordenadasIncorrectasException {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 6, 2 }, { 6, 5 }, { 6, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 6, 5, 6, 4));
		}

		/**
		 * Coloca piezas rodeando al rey de dos atacantes en horizontal en borde
		 * izquierdo.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 A - - - - - -
		 * 4 R - - - - - -
		 * 3 - - - - - - -
		 * 2 A - - - - - - 
	 	* 1 - - - - - - -
	 	*   a b c d e f g   
	 	*/
		// @formatter:on
		void rodearAlReyConDosAtacantesEnHorizontalEnBordeIzquierdo()  throws CoordenadasIncorrectasException {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 2, 0 }, { 5, 0 }, { 3, 0 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 5, 0, 4, 0));
		}

		/**
		 * Coloca piezas rodeando al rey de dos atacantes en horizontal en borde
		 * derecho.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - A 
		 * 5 - - - - - - -
		 * 4 - - - - - - R
		 * 3 - - - - - - A
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConDosAtacantesEnHorizontalEnBordeDerecho()  throws CoordenadasIncorrectasException {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 1, 6 }, { 4, 6 }, { 3, 6 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 1, 6, 2, 6));
		}
	}
	
	/**
	 * Victoria rodeando al rey con un atacante contra una provincia.
	 */
	@Nested
	@DisplayName("Tests para comprobar que victoria del atacante si rodea solo con un atacante y la provincia al rey.")
	class VictoriasRodeandoAlReyPorDosLadosConAtacanteYProvincia {

		/**
		 * Comprueba victoria rodeando al rey con una pieza atacante y la provincia en
		 * esquina superior izquierda en horizontal.
		 *
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarVictoriaRodeandoAlReyConAtacanteYProvinciaEnEsquinaSuperiorIzquierdaEnHorizontal()  throws CoordenadasIncorrectasException{
			rodearAlReyConAtacanteYProvinciaEnEsquinaSuperiorIzquierdaEnHorizontal();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con una pieza atacante y la provincia en
		 * esquina superior derecha en horizontal.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarVictoriaRodeandoAlReyConAtacanteYProvinciaEnEsquinaSuperiorDerechaEnHorizontal() throws CoordenadasIncorrectasException {
			rodearAlReyConAtacanteYProvinciaEnEsquinaSuperiorDerechaEnHorizontal();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con una pieza atacante y la provincia en
		 * esquina inferior izquierda en horizontal.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarVictoriaRodeandoAlReyConAtacanteYProvinciaEnEsquinaInferiorIzquierdaEnHorizontal()  throws CoordenadasIncorrectasException {
			rodearAlReyConAtacanteYProvinciaEnEsquinaInferiorIzquierdaEnHorizontal();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con una pieza atacante y la provincia en
		 * esquina inferior derecha en horizontal.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarVictoriaRodeandoAlReyConAtacanteYProvinciaEnEsquinaInferiorDerechaEnHorizontal()  throws CoordenadasIncorrectasException {
			rodearAlReyConAtacanteYProvinciaEnEsquinaInferiorDerechaEnHorizontal();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con una pieza atacante y la provincia en
		 * esquina superior izquierda en vertical.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarVictoriaRodeandoAlReyConAtacanteYProvinciaEnEsquinaSuperiorIzquierdaEnVertical()  throws CoordenadasIncorrectasException {
			rodearAlReyConAtacanteYProvinciaEnEsquinaSuperiorIzquierdaEnVertical();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con una pieza atacante y la provincia en
		 * esquina superior derecha en vertical.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarVictoriaRodeandoAlReyConAtacanteYProvinciaEnEsquinaSuperiorDerechaEnVertical()  throws CoordenadasIncorrectasException {
			rodearAlReyConAtacanteYProvinciaEnEsquinaSuperiorDerechaEnVertical();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con una pieza atacante y la provincia en
		 * esquina inferior izquierda en vertical.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarVictoriaRodeandoAlReyConAtacanteYProvinciaEnEsquinaInferiorIzquierdaEnVertical()  throws CoordenadasIncorrectasException {
			rodearAlReyConAtacanteYProvinciaEnEsquinaInferiorIzquierdaEnVertical();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con una pieza atacante y la provincia en
		 * esquina inferior derecha en vertical.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarVictoriaRodeandoAlReyConAtacanteYProvinciaEnEsquinaInferiorDerechaEnVertical()  throws CoordenadasIncorrectasException {
			rodearAlReyConAtacanteYProvinciaEnEsquinaInferiorDerechaEnVertical();
			comprobarReyRodeado();
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y provincia en horizontal en
		 * esquina superior izquierda.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - R - A - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - -
		 * 4 - - - - - - -
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConAtacanteYProvinciaEnEsquinaSuperiorIzquierdaEnHorizontal()  throws CoordenadasIncorrectasException {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 0, 3 }, { 0, 1 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 0, 3, 0, 2));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y provincia en horizontal en
		 * esquina superior derecha.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - A - R -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - -
		 * 4 - - - - - - -
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConAtacanteYProvinciaEnEsquinaSuperiorDerechaEnHorizontal() throws CoordenadasIncorrectasException {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 0, 3 }, { 0, 5 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 0, 3, 0, 4));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y provincia en horizontal en
		 * esquina inferior izquierda.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - -
		 * 4 - - - - - - -
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - R - A - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConAtacanteYProvinciaEnEsquinaInferiorIzquierdaEnHorizontal()  throws CoordenadasIncorrectasException{
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 6, 3 }, { 6, 1 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 6, 3, 6, 2));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y provincia en horizontal en
		 * esquina inferior derecha.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - -
		 * 4 - - - - - - -
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - A - R -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConAtacanteYProvinciaEnEsquinaInferiorDerechaEnHorizontal()  throws CoordenadasIncorrectasException{
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 6, 3 }, { 6, 5 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 6, 3, 6, 4));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y provincia en vertical en
		 * esquina superior izquierda.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 R - - - - - - 
		 * 5 - - - - - - -
		 * 4 A - - - - - -
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConAtacanteYProvinciaEnEsquinaSuperiorIzquierdaEnVertical()  throws CoordenadasIncorrectasException{
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 3, 0 }, { 1, 0 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 0, 2, 0));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y provincia en vertical en
		 * esquina superior derecha.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - R 
		 * 5 - - - - - - -
		 * 4 - - - - - - A
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConAtacanteYProvinciaEnEsquinaSuperiorDerechaEnVertical()  throws CoordenadasIncorrectasException{
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 3, 6 }, { 1, 6 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 6, 2, 6));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y provincia en vertical en
		 * esquina inferior izquierda.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - -
		 * 5 - - - - - - -
		 * 4 A - - - - - -
		 * 3 - - - - - - -
		 * 2 R - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConAtacanteYProvinciaEnEsquinaInferiorIzquierdaEnVertical()  throws CoordenadasIncorrectasException{
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 3, 0 }, { 5, 0 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 0, 4, 0));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y provincia en vertical en
		 * esquina inferior derecha.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - -
		 * 5 - - - - - - -
		 * 4 - - - - - - A
		 * 3 - - - - - - -
		 * 2 - - - - - - R 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConAtacanteYProvinciaEnEsquinaInferiorDerechaEnVertical()  throws CoordenadasIncorrectasException{
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 3, 6 }, { 5, 6 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 6, 4, 6));
		}
	}

}
