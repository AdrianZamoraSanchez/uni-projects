package tafl.control.medio;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Timeout.ThreadMode.SEPARATE_THREAD;
import static tafl.control.TestUtil.fabricarJugada;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import tafl.control.Arbitro;
import tafl.excepcion.CoordenadasIncorrectasException;
import tafl.modelo.Tablero;
import tafl.util.Color;
import tafl.util.TipoPieza;

/**
 * Comprobación de victoria de atacante rodeando al rey.
 * Aplicable a ambos tipos de juego.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0.0.1
 */
@DisplayName("Tests del arbitro sobre victorias del atacante rodeando al rey.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public abstract class ArbitroAbstractoVictoriaAtacanteTest {

	/** Árbitro de testing. */
	protected Arbitro arbitro;

	/** Tablero de testing. */
	protected Tablero tablero;

	/**
	 * Comprueba las condiciones cuando el rey está rodeado de piezas atacantes.
	 */
	protected void comprobarReyRodeado() {
		assertAll("victoria rodeando al rey",
				() -> assertThat("Debería ganar el atacante.", arbitro.haGanadoAtacante(), is(true)),
				() -> assertThat("No debería ganar el defensor.", arbitro.haGanadoRey(), is(false)),
				() -> assertThat("El turno debería ser del jugado atacante.", arbitro.consultarTurno(),
						is(Color.NEGRO)));
	}

	/**
	 * Comprueba las condiciones de no victoria cuando el rey no está rodeado usando
	 * el trono.
	 */
	protected void comprobarNoVictoria() {
		assertAll("no victoria rodeando al rey contra trono",
				() -> assertThat("No debería ganar el atacante.", arbitro.haGanadoAtacante(), is(false)),
				() -> assertThat("No debería ganar el defensor.", arbitro.haGanadoRey(), is(false)),
				() -> assertThat("El turno debería ser del jugado atacante.", arbitro.consultarTurno(),
						is(Color.NEGRO)));
	}

	/**
	 * No victoria de atacante rodeando al rey contra el trono.
	 * Aplicable a ambos tipos de juego.
	 */
	@Nested
	@DisplayName("Tests para comprobar que no hay victoria del atacante si rodea solo con un atacante y el trono al rey.")

	class NoVictoriaSiSoloSeRodeaAlReyConUnAtacanteYTrono {

		/**
		 * Comprueba que no hay victoria rodeando al rey con una pieza atacante y el
		 * trono al oeste sin más.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarNoVictoriaRodeandoAlReyConAtacanteYTronoAlOeste()  throws CoordenadasIncorrectasException {
			rodearAlReyConUnAtacanteYTronoEnOeste();
			comprobarNoVictoria();
		}

		/**
		 * Comprueba que no hay victoria rodeando al rey con una pieza atacante y el
		 * trono al este sin más.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarNoVictoriaRodeandoAlReyConAtacanteYTronoAlEste()  throws CoordenadasIncorrectasException {
			rodearAlReyConUnAtacanteYTronoEnEste();
			comprobarNoVictoria();
		}

		/**
		 * Comprueba que no hay victoria rodeando al rey con una pieza atacante y el
		 * trono al norte sin más.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarNoVictoriaRodeandoAlReyConAtacanteYTronoAlNorte() throws CoordenadasIncorrectasException {
			rodearAlReyConUnAtacanteYTronoEnNorte();
			comprobarNoVictoria();
		}

		/**
		 * Comprueba que no hay victoria rodeando al rey con una pieza atacante y el
		 * trono al sur sin más.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarNoVictoriaRodeandoAlReyConAtacanteYTronoAlSur()  throws CoordenadasIncorrectasException {
			rodearAlReyConUnAtacanteYTronoEnSur();
			comprobarNoVictoria();
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y trono al oeste.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
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
		void rodearAlReyConUnAtacanteYTronoEnOeste()  throws CoordenadasIncorrectasException {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 3, 6 }, { 3, 4 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 6, 3, 5));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y trono al este.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
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
		void rodearAlReyConUnAtacanteYTronoEnEste()  throws CoordenadasIncorrectasException {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 3, 0 }, { 3, 2 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 0, 3, 1));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y trono al norte.
		 * 
		 * @version 1.0.0.1
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
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
		void rodearAlReyConUnAtacanteYTronoEnNorte()  throws CoordenadasIncorrectasException {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 6, 3 }, { 4, 3 } }, Color.NEGRO); // Bug corregido
			// necesitamos mover para que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 6, 3, 5, 3));
		}

		/**
		 * Coloca pieza rodeando al rey con un atacante y trono al sur.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
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
		void rodearAlReyConUnAtacanteYTronoEnSur()  throws CoordenadasIncorrectasException {
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
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarNoVictoriaRodeandoAlReySoloConDosAtacantesEnHorizontalEstandoEnElTrono()  throws CoordenadasIncorrectasException {
			rodearAlReySoloConDosAtacantesEnHorizontal();
			comprobarNoVictoria();
		}

		/**
		 * Comprueba que no hay victoria rodeando al rey solo con dos atacantes en vertical
		 * cuando está situado en el trono.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarNoVictoriaRodeandoAlReySoloConDosAtacantesEnVerticalEstandoEnElTrono()  throws CoordenadasIncorrectasException {
			rodearAlReySoloConDosATacantesEnVertical();
			comprobarNoVictoria();
		}

		/**
		 * Coloca dos atacantes rodeando al rey en horizontal.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
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
		void rodearAlReySoloConDosAtacantesEnHorizontal()  throws CoordenadasIncorrectasException {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 3, 2 }, {3, 5}, { 3, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 5, 3, 4));
		}

		/**
		 * Coloca dos atacantes rodeando al rey en vertical.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
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
		void rodearAlReySoloConDosATacantesEnVertical()  throws CoordenadasIncorrectasException {
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 2, 3 }, {5, 3}, { 3, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 5, 3, 4, 3));
		}
	}
	
	/**
	 * Victoria de atacante rodeando al rey por las cuatro celdas contiguas cuando está en el trono.
	 * 
	 * @version 1.0.0.1
	 */
	@Nested
	@DisplayName("Tests para comprobar victoria del atacante si rodea con cuatro atacantes al rey cuando está en el trono.")
	class VictoriasRodeandoAlReyPorCuatroLadosConCuatroAtacantes {
		
		/**
		 * Comprueba victoria rodeando al rey en el trono con cuatro atacantes por el norte.
		 * @throws CoordenadasIncorrectasException 
		 * @throws IllegalArgumentException 
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por cuatro atacantes moviendo por el norte.")
		void comprobarVictoriaRodeandoAlReyEnElTronoAlNorte() throws IllegalArgumentException, CoordenadasIncorrectasException {
			rodearAlReyConCuatroAtacantesAlNorte();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey en el trono con cuatro atacantes por el sur.
		 * @throws CoordenadasIncorrectasException 
		 * @throws IllegalArgumentException 
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por cuatro atacantes moviendo por el sur.")
		void comprobarVictoriaRodeandoAlReyEnElTronoAlSur() throws IllegalArgumentException, CoordenadasIncorrectasException {
			rodearAlReyConCuatroAtacantesAlSur();
			comprobarReyRodeado();
		}
		
		/**
		 * Comprueba victoria rodeando al rey en el trono con cuatro atacantes por el este.
		 * @throws CoordenadasIncorrectasException 
		 * @throws IllegalArgumentException 
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por cuatro atacantes moviendo por el este.")
		void comprobarVictoriaRodeandoAlReyEnElTronoAlEste() throws IllegalArgumentException, CoordenadasIncorrectasException {
			rodearAlReyConCuatroAtacantesAlEste();
			comprobarReyRodeado();
		}
		
		/**
		 * Comprueba victoria rodeando al rey en el trono con cuatro atacantes por el oeste.
		 * @throws CoordenadasIncorrectasException 
		 * @throws IllegalArgumentException 
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por cuatro atacantes moviendo por el oeste.")
		void comprobarVictoriaRodeandoAlReyEnElTronoAlOeste() throws IllegalArgumentException, CoordenadasIncorrectasException {
			rodearAlReyConCuatroAtacantesAlOeste();
			comprobarReyRodeado();
		}
		
		/**
		 * Coloca piezas rodeando al rey de atacantes cerrando por el norte.
		 * @throws CoordenadasIncorrectasException 
		 * @throws IllegalArgumentException 
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
		void rodearAlReyConCuatroAtacantesAlNorte() throws IllegalArgumentException, CoordenadasIncorrectasException {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE,
							TipoPieza.REY },
					new int[][] { { 1, 3 }, { 3, 2 }, { 3, 4 }, { 4, 3 }, { 3, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 1, 3, 2, 3));
		}

		/**
		 * Coloca piezas rodeando al rey de atacantes cerrando por el sur.
		 * @throws CoordenadasIncorrectasException 
		 * @throws IllegalArgumentException 
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
		void rodearAlReyConCuatroAtacantesAlSur() throws IllegalArgumentException, CoordenadasIncorrectasException {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE,
							TipoPieza.REY },
					new int[][] { { 2, 3 }, { 3, 2 }, { 3, 4 }, { 5, 3 }, { 3, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 5, 3, 4, 3));
		}
		
		/**
		 * Coloca piezas rodeando al rey de atacantes cerrando por el este.
		 * @throws CoordenadasIncorrectasException 
		 * @throws IllegalArgumentException 
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
		void rodearAlReyConCuatroAtacantesAlEste() throws IllegalArgumentException, CoordenadasIncorrectasException {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE,
							TipoPieza.REY },
					new int[][] { { 2, 3 }, { 3, 2 }, { 3, 5 }, { 4, 3 }, { 3, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 5, 3, 4));
		}
		
		/**
		 * Coloca piezas rodeando al rey de atacantes cerrando por el oeste.
		 * @throws CoordenadasIncorrectasException 
		 * @throws IllegalArgumentException 
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
		void rodearAlReyConCuatroAtacantesAlOeste() throws IllegalArgumentException, CoordenadasIncorrectasException {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE,
							TipoPieza.REY },
					new int[][] { { 2, 3 }, { 3, 1 }, { 3, 4 }, { 4, 3 }, { 3, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 1, 3, 2));
		}
	}
	
	/**
	 * Victoria de atacante rodeando al rey por tres atacantes y el trono 
	 * cuando está situado fuera del trono y contiguo al mismo.
	 * 
	 * @version 1.0.0.1
	 */
	@Nested
	@DisplayName("Tests para comprobar victoria del atacante si rodea con tres atacantes y el trono al rey.")
	class VictoriasRodeandoAlReyPorTresAtacantesYTrono {	
		

		/**
		 * Comprueba victoria rodeando al rey con tres piezas atacantes y el trono al
		 * norte.
		 * @throws CoordenadasIncorrectasException 
		 * @throws IllegalArgumentException 
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por tres atacantes y el trono por el norte.")
		void comprobarVictoriaRodeandoAlReyConTresPiezasYLaCeldaDelTronoAlNorte() throws IllegalArgumentException, CoordenadasIncorrectasException {
			rodearAlReyConTresAtacantesYCeldaDelTronoAlNorte();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con tres piezas atacantes y el trono al
		 * oeste.
		 * @throws CoordenadasIncorrectasException 
		 * @throws IllegalArgumentException 
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por tres atacantes y el trono por el oeste.")
		void comprobarVictoriaRodeandoAlReyConTresPiezasYLaCeldaDelTronoAlOeste() throws IllegalArgumentException, CoordenadasIncorrectasException {
			rodearAlReyConTresAtacantesYCeldaDelTronoAlOeste();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con tres piezas atacantes y el trono al
		 * este.
		 * @throws CoordenadasIncorrectasException 
		 * @throws IllegalArgumentException 
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por tres atacantes y el trono por el este.")
		void comprobarVictoriaRodeandoAlReyConTresPiezasYLaCeldaDelTronoAlEste() throws IllegalArgumentException, CoordenadasIncorrectasException {
			rodearAlReyConTresAtacantesYCeldaDelTronoAlEste();
			comprobarReyRodeado();
		}

		/**
		 * Comprueba victoria rodeando al rey con tres piezas atacantes y el trono al
		 * sur.
		 * @throws CoordenadasIncorrectasException 
		 * @throws IllegalArgumentException 
		 */
		@Test
		@DisplayName("Victoria si el rey es rodeado por tres atacantes y el trono por el sur.")
		void comprobarVictoriaRodeandoAlReyConTresPiezasYLaCeldaDelTronoAlSur() throws IllegalArgumentException, CoordenadasIncorrectasException {
			rodearAlReyConTresAtacantesYCeldaDelTronoAlSur();
			comprobarReyRodeado();
		}
		
		
		/**
		 * Coloca piezas rodeando al rey de tres atacantes y la celda del trono.
		 * @throws CoordenadasIncorrectasException 
		 * @throws IllegalArgumentException 
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
		void rodearAlReyConTresAtacantesYCeldaDelTronoAlNorte() throws IllegalArgumentException, CoordenadasIncorrectasException {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 4, 2 }, { 4, 4 }, { 6, 3 }, { 4, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 6, 3, 5, 3));
		}

		/**
		 * Coloca piezas rodeando al rey de tres atacantes y la celda del trono al
		 * oeste.
		 * @throws CoordenadasIncorrectasException 
		 * @throws IllegalArgumentException 
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
		void rodearAlReyConTresAtacantesYCeldaDelTronoAlOeste() throws IllegalArgumentException, CoordenadasIncorrectasException {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 2, 4 }, { 3, 6 }, { 4, 4 }, { 3, 4 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 6, 3, 5));
		}

		/**
		 * Coloca piezas rodeando al rey de tres atacantes y la celda del trono al este.
		 * @throws CoordenadasIncorrectasException 
		 * @throws IllegalArgumentException 
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
		void rodearAlReyConTresAtacantesYCeldaDelTronoAlEste() throws IllegalArgumentException, CoordenadasIncorrectasException {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 2, 2 }, { 3, 0 }, { 4, 2 }, { 3, 2 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 0, 3, 1));
		}

		

		/**
		 * Coloca piezas rodeando al rey de tres atacantes y la celda del trono al sur.
		 * @throws CoordenadasIncorrectasException 
		 * @throws IllegalArgumentException 
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
		void rodearAlReyConTresAtacantesYCeldaDelTronoAlSur() throws IllegalArgumentException, CoordenadasIncorrectasException {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.REY },
					new int[][] { { 0, 3 }, { 2, 2 }, { 2, 4 }, { 2, 3 } }, Color.NEGRO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 0, 3, 1, 3));
		}
	}

}
