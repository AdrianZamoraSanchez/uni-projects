package tafl.control.ardri.medio;

import static org.junit.jupiter.api.Timeout.ThreadMode.SEPARATE_THREAD;
import static tafl.control.TestUtil.fabricarJugada;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import tafl.control.ArbitroArdRi;
import tafl.excepcion.CoordenadasIncorrectasException;
import tafl.modelo.Tablero;
import tafl.util.Color;
import tafl.util.TipoPieza;

/**
 * Comprobación con ArbitroArdRi de victoria de atacante rodeando al rey.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@DisplayName("Tests del ArbitroArdRi sobre victorias del atacante rodeando al rey.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public class ArbitroVictoriaAtacanteTest extends tafl.control.medio.ArbitroAbstractoVictoriaAtacanteTest {

	/** Generación del árbitro para testing. */
	@BeforeEach
	void inicializar() {
		// Inyección de tablero para testing...
		tablero = new Tablero();
		arbitro = new ArbitroArdRi(tablero);
	}
	
	/**
	 * Victoria de atacantes rodeando al rey con dos atacantes sin estar situado el
	 * trono.
	 */
	@Nested
	@DisplayName("Tests para comprobar que hay victoria del atacante si rodea con dos atacantes al rey cuando no está en el trono ni contiguo a él.")

	class VictoriasRodeandoAlReyPorDosLadosConAtacantesSinEstarEnElTronoOContiguoAlTrono {

		/**
		 * Comprueba victoria rodeando al rey con dos piezas atacantes cerca del borde
		 * superior.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarVictoriaRodeandoAlReyPorDosLadosEnHorizontalCercaDelBordeSuperior()  throws CoordenadasIncorrectasException {
			rodearAlReyConDosAtacantesEnHorizontalCercaDelBordeSuperior();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con dos piezas atacantes cerca del borde
		 * inferior.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarVictoriaRodeandoAlReyPorDosLadosEnHorizontalCercaDelBordeInferior()  throws CoordenadasIncorrectasException {
			rodearAlReyConDosAtacantesEnHorizontalCercaDelBordeInferior();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con dos piezas atacantes cerca del borde
		 * izquierdo.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarVictoriaRodeandoAlReyPorDosLadosEnHorizontalCercaDelBordeIzquierdo()  throws CoordenadasIncorrectasException {
			rodearAlReyConDosAtacantesEnHorizontalCercaDelBordeIzquierdo();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con dos piezas atacantes cerca del borde
		 * derecho.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarVictoriaRodeandoAlReyPorDosLadoEnHorizontalCercaDelBordeDerecho()  throws CoordenadasIncorrectasException {
			rodearAlReyConDosAtacantesEnHorizontalCercaDelBordeDerecho();
			comprobarReyRodeado();
		}

		/**
		 * Coloca piezas rodeando al rey de dos atacantes en horizontal cerca del borde
		 * superior.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - A - R A - - 
		 * 5 - - - - - - -
		 * 4 - - - - - - -
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConDosAtacantesEnHorizontalCercaDelBordeSuperior()  throws CoordenadasIncorrectasException {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 1, 1 }, { 1, 4 }, { 1, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 1, 1, 1, 2));
		}

		/**
		 * Coloca piezas rodeando al rey de dos atacantes en horizontal cerca del borde
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
		 * 2 - - A R - A - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConDosAtacantesEnHorizontalCercaDelBordeInferior()  throws CoordenadasIncorrectasException {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 5, 2 }, { 5, 5 }, { 5, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 5, 5, 5, 4));
		}

		/**
		 * Coloca piezas rodeando al rey de dos atacantes en horizontal cerca del borde
		 * izquierdo.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - A - - - - -
		 * 4 - R - - - - -
		 * 3 - - - - - - -
		 * 2 - A - - - - - 
	 	* 1 - - - - - - -
	 	*   a b c d e f g   
	 	*/
		// @formatter:on
		void rodearAlReyConDosAtacantesEnHorizontalCercaDelBordeIzquierdo()  throws CoordenadasIncorrectasException {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 2, 1 }, { 5, 1 }, { 3, 1 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 5, 1, 4, 1));
		}

		/**
		 * Coloca piezas rodeando al rey de dos atacantes en horizontal cerca del borde
		 * derecho.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - A - 
		 * 5 - - - - - - -
		 * 4 - - - - - R -
		 * 3 - - - - - A -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConDosAtacantesEnHorizontalCercaDelBordeDerecho()  throws CoordenadasIncorrectasException {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 1, 5 }, { 4, 5 }, { 3, 5 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 1, 5, 2, 5));
		}
	}
}