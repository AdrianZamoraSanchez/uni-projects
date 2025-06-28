package brandubh.control.medio;

import static brandubh.control.TestUtil.fabricarJugada;
import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertAll;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import brandubh.control.Arbitro;
import brandubh.modelo.Tablero;
import brandubh.util.Color;
import brandubh.util.TipoPieza;

/**
 * Comprobación de victoria de atacante rodeando al rey.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@DisplayName("Tests del Arbitro sobre victorias del atacante rodeando al rey.")
@Timeout(value = 2, unit = TimeUnit.SECONDS) // Time out global para todos los tests salvo los de ciclo de vida
public class ArbitroVictoriaAtacanteTest {

	/** Árbitro de testing. */
	private Arbitro arbitro;

	/** Tablero de testing. */
	private Tablero tablero;

	/** Generación del árbitro para testing. */
	@BeforeEach
	void inicializar() {
		// Inyección de tablero para testing...
		tablero = new Tablero();
		arbitro = new Arbitro(tablero);
	}

	/**
	 * Comprueba las condiciones cuando el rey está rodeado de piezas atacantes.
	 */
	private void comprobarReyRodeado() {
		assertAll("victoria rodeando al rey",
				() -> assertThat("Debería ganar el atacante.", arbitro.haGanadoAtacante(), is(true)),
				() -> assertThat("No debería ganar el defensor.", arbitro.haGanadoRey(), is(false)));
	}

	/**
	 * Comprueba las condiciones de no victoria cuando el rey no está rodeado usando
	 * el trono.
	 */
	private void comprobarNoVictoriaDelAtacante() {
		assertAll("no victoria rodeando al rey contra trono",
				() -> assertThat("No debería ganar el atacante.", arbitro.haGanadoAtacante(), is(false)),
				() -> assertThat("No debería ganar el defensor.", arbitro.haGanadoRey(), is(false)));
	}	
	
	
	/**
	 * Victoria de atacante rodeando al rey por tres atacantes y el trono 
	 * cuando está situado fuera del trono y contiguo al mismo.
	 */
	@Nested
	@DisplayName("Tests para comprobar victoria del atacante si rodea con tres atacantes y el trono al rey.")
	class VictoriasRodeandoAlReyPorTresAtacantesYTrono {
		
		

		/**
		 * Comprueba victoria rodeando al rey con tres piezas atacantes y el trono al
		 * norte.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por tres atacantes y el trono por el norte.")
		void comprobarVictoriaRodeandoAlReyConTresPiezasYLaCeldaDelTronoAlNorte() {
			rodearAlReyConTresAtacantesYCeldaDelTronoAlNorte();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con tres piezas atacantes y el trono al
		 * oeste.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por tres atacantes y el trono por el oeste.")
		void comprobarVictoriaRodeandoAlReyConTresPiezasYLaCeldaDelTronoAlOeste() {
			rodearAlReyConTresAtacantesYCeldaDelTronoAlOeste();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con tres piezas atacantes y el trono al
		 * este.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por tres atacantes y el trono por el este.")
		void comprobarVictoriaRodeandoAlReyConTresPiezasYLaCeldaDelTronoAlEste() {
			rodearAlReyConTresAtacantesYCeldaDelTronoAlEste();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con tres piezas atacantes y el trono al
		 * sur.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por tres atacantes y el trono por el sur.")
		void comprobarVictoriaRodeandoAlReyConTresPiezasYLaCeldaDelTronoAlSur() {
			rodearAlReyConTresAtacantesYCeldaDelTronoAlSur();
			comprobarReyRodeado();
		}
		
		
		/**
		 * Coloca piezas rodeando al rey de tres atacantes y la celda del trono.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - - 
		 * 4 - - - - - - -
		 * 3 - - A R A - -
		 * 2 - - - - - - - 
		 * 1 - - - A - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		void rodearAlReyConTresAtacantesYCeldaDelTronoAlNorte() {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 4, 2 }, { 4, 4 }, { 6, 3 }, { 4, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 6, 3, 5, 3));
		}

		/**
		 * Coloca piezas rodeando al rey de tres atacantes y la celda del trono al
		 * oeste.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - A - -
		 * 4 - - - - R - A
		 * 3 - - - - A - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		void rodearAlReyConTresAtacantesYCeldaDelTronoAlOeste() {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 2, 4 }, { 3, 6 }, { 4, 4 }, { 3, 4 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 6, 3, 5));
		}

		/**
		 * Coloca piezas rodeando al rey de tres atacantes y la celda del trono al este.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - A - - - -
		 * 4 A - R - - - -
		 * 3 - - A - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		void rodearAlReyConTresAtacantesYCeldaDelTronoAlEste() {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 2, 2 }, { 3, 0 }, { 4, 2 }, { 3, 2 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 0, 3, 1));
		}

		

		/**
		 * Coloca piezas rodeando al rey de tres atacantes y la celda del trono al sur.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - A - - -  
		 * 6 - - - - - - - 
		 * 5 - - A R A - -
		 * 4 - - - - - - -
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		void rodearAlReyConTresAtacantesYCeldaDelTronoAlSur() {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 0, 3 }, { 2, 2 }, { 2, 4 }, { 2, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 0, 3, 1, 3));
		}
	}
	
	
	/**
	 * Victoria de atacante rodeando al rey por las cuatro celdas contiguas cuando está en el trono.
	 */
	@Nested
	@DisplayName("Tests para comprobar victoria del atacante si rodea con cuatro atacantes al rey cuando está en el trono.")
	class VictoriasRodeandoAlReyPorCuatroLadosConCuatroAtacantes {
		
		/**
		 * Comprueba victoria rodeando al rey en el trono con cuatro atacantes por el norte.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por cuatro atacantes moviendo por el norte.")
		void comprobarVictoriaRodeandoAlReyEnElTronoAlNorte() {
			rodearAlReyConCuatroAtacantesAlNorte();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey en el trono con cuatro atacantes por el sur.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por cuatro atacantes moviendo por el sur.")
		void comprobarVictoriaRodeandoAlReyEnElTronoAlSur() {
			rodearAlReyConCuatroAtacantesAlSur();
			comprobarReyRodeado();
		}
		
		/**
		 * Comprueba victoria rodeando al rey en el trono con cuatro atacantes por el este.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por cuatro atacantes moviendo por el este.")
		void comprobarVictoriaRodeandoAlReyEnElTronoAlEste() {
			rodearAlReyConCuatroAtacantesAlEste();
			comprobarReyRodeado();
		}
		
		/**
		 * Comprueba victoria rodeando al rey en el trono con cuatro atacantes por el oeste.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por cuatro atacantes moviendo por el oeste.")
		void comprobarVictoriaRodeandoAlReyEnElTronoAlOeste() {
			rodearAlReyConCuatroAtacantesAlOeste();
			comprobarReyRodeado();
		}
		
		/**
		 * Coloca piezas rodeando al rey de atacantes cerrando por el norte.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - A - - - 
		 * 5 - - - - - - - 
		 * 4 - - A R A - -
		 * 3 - - - A - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		void rodearAlReyConCuatroAtacantesAlNorte() {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE,
							TipoPieza.REY },
					new int[][] { { 1, 3 }, { 3, 2 }, { 3, 4 }, { 4, 3 }, { 3, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 1, 3, 2, 3));
		}

		/**
		 * Coloca piezas rodeando al rey de atacantes cerrando por el sur.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - A - - - 
		 * 4 - - A R A - -
		 * 3 - - - - - - -
		 * 2 - - - A - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		void rodearAlReyConCuatroAtacantesAlSur() {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE,
							TipoPieza.REY },
					new int[][] { { 2, 3 }, { 3, 2 }, { 3, 4 }, { 5, 3 }, { 3, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 5, 3, 4, 3));
		}
		
		/**
		 * Coloca piezas rodeando al rey de atacantes cerrando por el este.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - A - - - 
		 * 4 - - A R - A -
		 * 3 - - - A - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		void rodearAlReyConCuatroAtacantesAlEste() {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE,
							TipoPieza.REY },
					new int[][] { { 2, 3 }, { 3, 2 }, { 3, 5 }, { 4, 3 }, { 3, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 5, 3, 4));
		}
		
		/**
		 * Coloca piezas rodeando al rey de atacantes cerrando por el oeste.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - A - - - 
		 * 4 - A - R A - -
		 * 3 - - - A - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		void rodearAlReyConCuatroAtacantesAlOeste() {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE,
							TipoPieza.REY },
					new int[][] { { 2, 3 }, { 3, 1 }, { 3, 4 }, { 4, 3 }, { 3, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 1, 3, 2));
		}
	}
	
	
	/**
	 * Victoria de atacantes rodeando al rey con dos atacantes sin estar situado en el
	 * trono ni contiguo a él en los bordes del tablero.
	 */
	@Nested
	@DisplayName("Tests para comprobar que hay victoria del atacante si rodea con dos atacantes al rey cuando no está en el trono ni contiguo a él en los bordes del tablero.")

	class VictoriasRodeandoAlReyPorDosLadosConAtacantesSinEstarEnElTronoOContiguoAlTronoEnBordesDelTablero {

		/**
		 * Comprueba victoria rodeando al rey con dos piezas atacantes en horizontal en el borde
		 * superior.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por dos atacantes en horizontal en borde superior.")
		void comprobarVictoriaRodeandoAlReyPorDosLadosEnHorizontalEnBordeSuperior() {
			rodearAlReyConDosAtacantesEnHorizontalEnBordeSuperior();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con dos piezas atacantes en horizontal en el borde
		 * inferior.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por dos atacantes en horizontal en borde inferior.")
		void comprobarVictoriaRodeandoAlReyPorDosLadosEnHorizontalEnBordeInferior() {
			rodearAlReyConDosAtacantesEnHorizontalEnBordeInferior();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con dos piezas atacantes en horizontal en el borde
		 * izquierdo.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por dos atacantes en horizontal en borde izquierdo.")
		void comprobarVictoriaRodeandoAlReyPorDosLadosEnHorizontalEnBordeIzquierdo() {
			rodearAlReyConDosAtacantesEnHorizontalEnBordeIzquierdo();
			comprobarReyRodeado();
		}
		
		/**
		 * Comprueba victoria rodeando al rey con dos piezas atacantes en horizontal en el borde
		 * derecho.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por dos atacantes en horizontal en borde derecho.")
		void comprobarVictoriaRodeandoAlReyPorDosLadosEnHorizontalEnBordeDerecho() {
			rodearAlReyConDosAtacantesEnHorizontalEnBordeDerecho();
			comprobarReyRodeado();
		}
		
		/**
		 * Comprueba victoria rodeando al rey con dos piezas atacantes en vertical en el borde
		 * superior.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por dos atacantes en vertical en borde superior.")
		void comprobarVictoriaRodeandoAlReyPorDosLadosEnVerticalEnBordeSuperior() {
			rodearAlReyConDosAtacantesEnVerticalEnBordeSuperior();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con dos piezas atacantes en vertical en el borde
		 * inferior.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por dos atacantes en vertical en borde inferior.")
		void comprobarVictoriaRodeandoAlReyPorDosLadosEnVerticalEnBordeInferior() {
			rodearAlReyConDosAtacantesEnVerticalEnBordeInferior();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con dos piezas atacantes en vertical en el borde
		 * izquierdo.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por dos atacantes en vertical en borde izquierdo.")
		void comprobarVictoriaRodeandoAlReyPorDosLadosEnVerticalEnBordeIzquierdo() {
			rodearAlReyConDosAtacantesEnVerticalEnBordeIzquierdo();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con dos piezas atacantes en vertical en el borde
		 * derecho.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por dos atacantes en horizontal en borde derecho.")
		void comprobarVictoriaRodeandoAlReyPorDosLadosEnVerticalEnBordeDerecho() {
			rodearAlReyConDosAtacantesEnVerticalEnBordeDerecho();
			comprobarReyRodeado();
		}

		/**
		 * Coloca piezas rodeando al rey de dos atacantes en horizontal en borde
		 * superior.
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
		void rodearAlReyConDosAtacantesEnHorizontalEnBordeSuperior() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 0, 1 }, { 0, 4 }, { 0, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 0, 1, 0, 2));
		}

		/**
		 * Coloca piezas rodeando al rey de dos atacantes en horizontal en borde
		 * inferior.
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
		void rodearAlReyConDosAtacantesEnHorizontalEnBordeInferior() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 6, 2 }, { 6, 5 }, { 6, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 6, 5, 6, 4));
		}

		/**
		 * Coloca piezas rodeando al rey de dos atacantes en horizontal en borde
		 * izquierdo.
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
		void rodearAlReyConDosAtacantesEnHorizontalEnBordeIzquierdo() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 2, 0 }, { 5, 0 }, { 3, 0 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 5, 0, 4, 0));
		}

		/**
		 * Coloca piezas rodeando al rey de dos atacantes en horizontal en borde
		 * derecho.
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
		void rodearAlReyConDosAtacantesEnHorizontalEnBordeDerecho() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 1, 6 }, { 4, 6 }, { 3, 6 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 1, 6, 2, 6));
		}
		
		
		/**
		 * Coloca piezas rodeando al rey de dos atacantes en vertical en borde
		 * superior.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - A - - - -  
		 * 6 - - R - - - - 
		 * 5 - - - - - - -
		 * 4 - - A - - - -
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConDosAtacantesEnVerticalEnBordeSuperior() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 0, 2}, { 3, 2}, { 1, 2 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 2, 2, 2));
		}
		
		/**
		 * Coloca piezas rodeando al rey de dos atacantes en vertical en borde
		 * inferior.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - -
		 * 4 - - - - - A -
		 * 3 - - - - - - -
		 * 2 - - - - - R - 
		 * 1 - - - - - A -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConDosAtacantesEnVerticalEnBordeInferior() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 3, 5 }, { 6, 5 }, { 5, 5 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 5, 4, 5));
		}
		
		/**
		 * Coloca piezas rodeando al rey de dos atacantes en vertical en borde
		 * izquierdo.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 A R - A - - -
		 * 4 - - - - - - -
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
	 	 * 1 - - - - - - -
	 	 *   a b c d e f g   
	 	 */
		// @formatter:on
		void rodearAlReyConDosAtacantesEnVerticalEnBordeIzquierdo() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 2, 0 }, { 2, 3 }, { 2, 1 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 2, 3, 2, 2));
		}
		
		/**
		 * Coloca piezas rodeando al rey de dos atacantes en vertical en borde
		 * derecho.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - -
		 * 4 - - - - - - -
		 * 3 - - - A - R A
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConDosAtacantesEnVerticalEnBordeDerecho() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 4, 3 }, { 4, 6 }, { 4, 5 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 4, 3, 4, 4));
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
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por atacante y  provincia en horizontal en esquina superior izquierda.")
		void comprobarVictoriaRodeandoAlReyConAtacanteYProvinciaEnEsquinaSuperiorIzquierdaEnHorizontal() {
			rodearAlReyConAtacanteYProvinciaEnEsquinaSuperiorIzquierdaEnHorizontal();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con una pieza atacante y la provincia en
		 * esquina superior derecha en horizontal.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por atacante y  provincia en horizontal en esquina superior derecha.")
		void comprobarVictoriaRodeandoAlReyConAtacanteYProvinciaEnEsquinaSuperiorDerechaEnHorizontal() {
			rodearAlReyConAtacanteYProvinciaEnEsquinaSuperiorDerechaEnHorizontal();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con una pieza atacante y la provincia en
		 * esquina inferior izquierda en horizontal.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por atacante y  provincia en horizontal en esquina inferior izquierda.")
		void comprobarVictoriaRodeandoAlReyConAtacanteYProvinciaEnEsquinaInferiorIzquierdaEnHorizontal() {
			rodearAlReyConAtacanteYProvinciaEnEsquinaInferiorIzquierdaEnHorizontal();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con una pieza atacante y la provincia en
		 * esquina inferior derecha en horizontal.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por atacante y  provincia en horizontal en esquina inferior derecha.")
		void comprobarVictoriaRodeandoAlReyConAtacanteYProvinciaEnEsquinaInferiorDerechaEnHorizontal() {
			rodearAlReyConAtacanteYProvinciaEnEsquinaInferiorDerechaEnHorizontal();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con una pieza atacante y la provincia en
		 * esquina superior izquierda en vertical.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por atacante y  provincia en vertical en esquina superior izquierda.")
		void comprobarVictoriaRodeandoAlReyConAtacanteYProvinciaEnEsquinaSuperiorIzquierdaEnVertical() {
			rodearAlReyConAtacanteYProvinciaEnEsquinaSuperiorIzquierdaEnVertical();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con una pieza atacante y la provincia en
		 * esquina superior derecha en vertical.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por atacante y  provincia en vertical en esquina superior derecha.")
		void comprobarVictoriaRodeandoAlReyConAtacanteYProvinciaEnEsquinaSuperiorDerechaEnVertical() {
			rodearAlReyConAtacanteYProvinciaEnEsquinaSuperiorDerechaEnVertical();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con una pieza atacante y la provincia en
		 * esquina inferior izquierda en vertical.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por atacante y  provincia en vertical en esquina inferior izquierda.")
		void comprobarVictoriaRodeandoAlReyConAtacanteYProvinciaEnEsquinaInferiorIzquierdaEnVertical() {
			rodearAlReyConAtacanteYProvinciaEnEsquinaInferiorIzquierdaEnVertical();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con una pieza atacante y la provincia en
		 * esquina inferior derecha en vertical.
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por atacante y  provincia en vertical en esquina inferior derecha.")
		void comprobarVictoriaRodeandoAlReyConAtacanteYProvinciaEnEsquinaInferiorDerechaEnVertical() {
			rodearAlReyConAtacanteYProvinciaEnEsquinaInferiorDerechaEnVertical();
			comprobarReyRodeado();
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y provincia en horizontal en
		 * esquina superior izquierda.
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
		void rodearAlReyConAtacanteYProvinciaEnEsquinaSuperiorIzquierdaEnHorizontal() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 0, 3 }, { 0, 1 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 0, 3, 0, 2));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y provincia en horizontal en
		 * esquina superior derecha.
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
		void rodearAlReyConAtacanteYProvinciaEnEsquinaSuperiorDerechaEnHorizontal() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 0, 3 }, { 0, 5 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 0, 3, 0, 4));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y provincia en horizontal en
		 * esquina inferior izquierda.
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
		void rodearAlReyConAtacanteYProvinciaEnEsquinaInferiorIzquierdaEnHorizontal() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 6, 3 }, { 6, 1 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 6, 3, 6, 2));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y provincia en horizontal en
		 * esquina inferior derecha.
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
		void rodearAlReyConAtacanteYProvinciaEnEsquinaInferiorDerechaEnHorizontal() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 6, 3 }, { 6, 5 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 6, 3, 6, 4));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y provincia en vertical en
		 * esquina superior izquierda.
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
		void rodearAlReyConAtacanteYProvinciaEnEsquinaSuperiorIzquierdaEnVertical() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 3, 0 }, { 1, 0 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 0, 2, 0));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y provincia en vertical en
		 * esquina superior derecha.
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
		void rodearAlReyConAtacanteYProvinciaEnEsquinaSuperiorDerechaEnVertical() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 3, 6 }, { 1, 6 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 6, 2, 6));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y provincia en vertical en
		 * esquina inferior izquierda.
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
		void rodearAlReyConAtacanteYProvinciaEnEsquinaInferiorIzquierdaEnVertical() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 3, 0 }, { 5, 0 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 0, 4, 0));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y provincia en vertical en
		 * esquina inferior derecha.
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
		void rodearAlReyConAtacanteYProvinciaEnEsquinaInferiorDerechaEnVertical() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 3, 6 }, { 5, 6 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 6, 4, 6));
		}
	}

	/**
	 * No victoria de atacante rodeando al rey contra el trono.
	 */
	@Nested
	@DisplayName("Tests para comprobar que no hay victoria del atacante si rodea solo con un atacante y el trono al rey.")
	class NoVictoriaSiSoloSeRodeaAlReyConUnAtacanteYTrono {

		/**
		 * Comprueba que no hay victoria rodeando al rey con una pieza atacante y el
		 * trono al oeste sin más.
		 */
		@Test
		@DisplayName("No victoria si el rey es rodeado por un atacante y el trono al oeste.")
		void comprobarNoVictoriaRodeandoAlReyConAtacanteYTronoAlOeste() {
			rodearAlReyConUnAtacanteYTronoEnOeste();
			comprobarNoVictoriaDelAtacante();
		}

		/**
		 * Comprueba que no hay victoria rodeando al rey con una pieza atacante y el
		 * trono al este sin más.
		 */
		@Test
		@DisplayName("No victoria si el rey es rodeado por un atacante y el trono al este.")
		void comprobarNoVictoriaRodeandoAlReyConAtacanteYTronoAlEste() {
			rodearAlReyConUnAtacanteYTronoEnEste();
			comprobarNoVictoriaDelAtacante();
		}

		/**
		 * Comprueba que no hay victoria rodeando al rey con una pieza atacante y el
		 * trono al norte sin más.
		 */
		@Test
		@DisplayName("No victoria si el rey es rodeado por un atacante y el trono al norte.")
		void comprobarNoVictoriaRodeandoAlReyConAtacanteYTronoAlNorte() {
			rodearAlReyConUnAtacanteYTronoEnNorte();
			comprobarNoVictoriaDelAtacante();
		}

		/**
		 * Comprueba que no hay victoria rodeando al rey con una pieza atacante y el
		 * trono al sur sin más.
		 */
		@Test
		@DisplayName("No victoria si el rey es rodeado por un atacante y el trono al sur.")
		void comprobarNoVictoriaRodeandoAlReyConAtacanteYTronoAlSur() {
			rodearAlReyConUnAtacanteYTronoEnSur();
			comprobarNoVictoriaDelAtacante();
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y trono al oeste.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - -
		 * 5 - - - - - - -
		 * 4 - - - - R - A
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConUnAtacanteYTronoEnOeste() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 3, 6 }, { 3, 4 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 6, 3, 5));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y trono al este.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - -
		 * 5 - - - - - - -
		 * 4 A - R - - - -
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConUnAtacanteYTronoEnEste() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 3, 0 }, { 3, 2 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 0, 3, 1));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y trono al norte.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - -
		 * 5 - - - - - - -
		 * 4 - - - - - - -
		 * 3 - - - R - - -
		 * 2 - - - - - - - 
		 * 1 - - - A - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConUnAtacanteYTronoEnNorte() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 4, 3 }, { 6, 3 } }, Color.NEGRO);
			// necesitamos mover para que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 6, 3, 5, 3));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y trono al sur.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - A - - -  
		 * 6 - - - - - - -
		 * 5 - - - R - - -
		 * 4 - - - - - - -
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReyConUnAtacanteYTronoEnSur() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 0, 3 }, { 2, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 0, 3, 1, 3));
		}
	}

	/**
	 * No victoria de atacante rodeando al rey solo con dos atacantes cuando está en el trono.
	 */
	@Nested
	@DisplayName("Tests para comprobar que no hay victoria del atacante si rodea solo con dos atacantes al rey cuando está en el trono.")
	class NoVictoriaSiSeRodeaAlReySoloConDosAtacantesCuandoEstaEnElTrono {

		/**
		 * Comprueba que no hay victoria rodeando al rey solo con dos atacantes en horizontal
		 * cuando está situado en el trono.
		 */
		@Test
		@DisplayName("No victoria si el rey es rodeado por solo dos atacantes en horizontal estando en el trono.")
		void comprobarNoVictoriaRodeandoAlReySoloConDosAtacantesEnHorizontalEstandoEnElTrono() {
			rodearAlReySoloConDosAtacantesEnHorizontal();
			comprobarNoVictoriaDelAtacante();
		}

		/**
		 * Comprueba que no hay victoria rodeando al rey solo con dos atacantes en vertical
		 * cuando está situado en el trono.
		 */
		@Test
		@DisplayName("No victoria si el rey es rodeado por solo dos atacantes en vertical estando en el trono.")
		void comprobarNoVictoriaRodeandoAlReySoloConDosAtacantesEnVerticalEstandoEnElTrono() {
			rodearAlReySoloConDosATacantesEnVertical();
			comprobarNoVictoriaDelAtacante();
		}

		/**
		 * Coloca dos atacantes rodeando al rey en horizontal.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - -
		 * 5 - - - - - - -
		 * 4 - - A R - A -
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReySoloConDosAtacantesEnHorizontal() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 3, 2 }, {3, 5}, { 3, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 5, 3, 4));
		}

		/**
		 * Coloca dos atacantes rodeando al rey en vertical.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - -
		 * 5 - - - A - - -
		 * 4 - - - R - - -
		 * 3 - - - - - - -
		 * 2 - - - A - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		void rodearAlReySoloConDosATacantesEnVertical() {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 2, 3 }, {5, 3}, { 3, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 5, 3, 4, 3));
		}
	}

}
