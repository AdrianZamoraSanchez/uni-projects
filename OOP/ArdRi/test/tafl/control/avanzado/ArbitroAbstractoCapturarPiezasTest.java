package tafl.control.avanzado;

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
import tafl.util.Coordenada;
import tafl.util.TipoPieza;

/**
 * Comprobación de captura de piezas en distintas condiciones.
 * Aplicable a ambos tipos de juego.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@DisplayName("Tests del arbitro sobre capturas de piezas.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public abstract class ArbitroAbstractoCapturarPiezasTest {

	/** Árbitro de testing. */
	protected Arbitro arbitro;

	/** Tablero de testing. */
	protected Tablero tablero;

	/**
	 * Captura de tres piezas en una sola jugada.
	 */
	@Nested
	@DisplayName("Tests de captura de tres piezas en una sola jugada.")
	class CapturarTresPiezasEnUnaSolaJugada {

		/**
		 * Comprueba que se capturan tres defensores en una sola jugada.
		 *
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - - 
		 * 4 - - A R - - - 
		 * 3 A D - D A - -
		 * 2 - - D - - - - 
		 * 1 - - A - - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		@Test
		void comprobarCapturaTresPiezasDefensorasEnUnaSolaJugada() throws CoordenadasIncorrectasException {
			// 4 atacantes, 3 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE,
							TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 4, 0 }, { 3, 2 }, { 4, 4 }, { 6, 2 }, { 4, 1 }, { 4, 3 }, { 5, 2 }, { 3, 3 } },
					Color.NEGRO);
			arbitro.mover(fabricarJugada(tablero, 3, 2, 4, 2)); // atacante
			arbitro.realizarCapturasTrasMover(); // elimina los tres defensores de golpe

			assertAll("captura de tres piezas defensoras en un solo movimiento",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(4, 1)).estaVacia(), is(true)),
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(4, 3)).estaVacia(), is(true)),
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(5, 2)).estaVacia(), is(true)),
					() -> assertThat("No debería cambiar el número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(4)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería capturarse tres piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(0)));
		}

		/**
		 * Comprueba que se capturan tres atacantes en una sola jugada.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - D - - 
		 * 4 - - - R A - - 
		 * 3 - - - D - A D
		 * 2 - - - - A - - 
		 * 1 - - - - D - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		@Test
		void comprobarCapturaTresPiezasAtacantesEnUnaSolaJugada() throws CoordenadasIncorrectasException {
			// 4 atacantes, 3 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.DEFENSOR,
							TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 3, 4 }, { 4, 5 }, { 5, 4 }, { 2, 4 }, { 4, 3 }, { 4, 6 }, { 6, 4 }, { 3, 3 } },
					Color.BLANCO);
			arbitro.mover(fabricarJugada(tablero, 4, 3, 4, 4)); // defensor
			arbitro.realizarCapturasTrasMover(); // elimina los tres atacantes de golpe

			assertAll("captura de tres piezas atacantes en un solo movimiento",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(3, 4)).estaVacia(), is(true)),
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(4, 5)).estaVacia(), is(true)),
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(5, 4)).estaVacia(), is(true)),
					() -> assertThat("Debería capturarse tres piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(0)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)));
		}

	}

	/**
	 * Captura de dos piezas en una sola jugada.
	 */
	@Nested
	@DisplayName("Tests de captura de dos piezas en una sola jugada.")
	class CapturarDosPiezasEnUnaSolaJugada {

		/**
		 * Comprueba que se capturan dos atacantes en una sola jugada.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - - - -  
		 * 6 D - - - - - - 
		 * 5 A - - - - - - 
		 * 4 - D - R - - - 
		 * 3 A - - - - - -
		 * 2 D - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarCapturaDosPiezasAtacantesEnUnaSolaJugada() throws CoordenadasIncorrectasException {
			// 2 atacantes, 3 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR,
							TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 2, 0 }, { 4, 0 }, { 1, 0 }, { 3, 1 }, { 5, 0 }, { 3, 3 } }, Color.BLANCO);
			arbitro.mover(fabricarJugada(tablero, 3, 1, 3, 0)); // defensor
			arbitro.realizarCapturasTrasMover(); // elimina los dos atacantes de golpe

			assertAll("captura de dos piezas atacantes en un solo movimiento",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(2, 0)).estaVacia(), is(true)),
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(4, 0)).estaVacia(), is(true)),
					() -> assertThat("Deberían capturarse dos piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(0)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(3)));
		}

		/**
		 * Comprueba que se capturan dos defensores en una sola jugada.
		 *
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - A D - D A -  
		 * 6 - - - A - - - 
		 * 5 - - - - - - - 
		 * 4 - - - R - - - 
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarCapturaDosPiezasDefensorasEnUnaSolaJugada() throws CoordenadasIncorrectasException {
			// 3 atacantes, 2 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.DEFENSOR,
							TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 0, 1 }, { 0, 5 }, { 1, 3 }, { 0, 2 }, { 0, 4 }, { 3, 3 } }, Color.NEGRO);
			arbitro.mover(fabricarJugada(tablero, 1, 3, 0, 3)); // atacante
			arbitro.realizarCapturasTrasMover(); // elimina los dos defensores de golpe

			assertAll("captura de dos piezas defensoras en un solo movimiento",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(0, 2)).estaVacia(), is(true)),
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(0, 4)).estaVacia(), is(true)),
					() -> assertThat("Deberían capturarse dos piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(0)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(3)));
		}

		/**
		 * Comprueba que se capturan dos atacantes en una sola jugada contra una provincia.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 *
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - A 
		 * 5 - - - - - D - 
		 * 4 - - - R - - A 
		 * 3 - - - - - - D
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarCapturaDosPiezasAtacantesEnUnSoloMovimientoContraProvincia() throws CoordenadasIncorrectasException {
			// 2 atacantes, 2 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR,
							TipoPieza.REY },
					new int[][] { { 1, 6 }, { 3, 6 }, { 2, 5 }, { 4, 6 }, { 3, 3 } }, Color.BLANCO);
			arbitro.mover(fabricarJugada(tablero, 2, 5, 2, 6)); // defensor
			arbitro.realizarCapturasTrasMover(); // elimina los dos atacantes de golpe

			assertAll("captura de dos piezas atacantes en un solo movimiento contra provincia",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(1, 6)).estaVacia(), is(true)),
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(3, 6)).estaVacia(), is(true)),
					() -> assertThat("Deberían capturarse dos piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(0)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(2)));
		}

		/**
		 * Comprueba que se capturan dos defensores en una sola jugada contra una provincia.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - -  - - 
		 * 4 - - - R - - - 
		 * 3 - - - - - - -
		 * 2 - - - - A - - 
		 * 1 - - A D - D -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarCapturaDosPiezasDefensorasEnUnSoloMovimientoContraProvincia() throws CoordenadasIncorrectasException {
			// 2 atacantes, 2 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR,
							TipoPieza.REY },
					new int[][] { { 5, 4 }, { 6, 2 }, { 6, 3 }, { 6, 5 }, { 3, 3 } }, Color.BLANCO);
			arbitro.mover(fabricarJugada(tablero, 5, 4, 6, 4)); // atacante
			arbitro.realizarCapturasTrasMover(); // elimina los dos defensores de golpe

			assertAll("captura de dos piezas defensoras en un solo movimiento contra provincia",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(6, 3)).estaVacia(), is(true)),
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(6, 5)).estaVacia(), is(true)),
					() -> assertThat("Deberían capturarse dos piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(0)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(2)));
		}
	}

	/**
	 * Captura de piezas contra el trono.
	 */
	@Nested
	@DisplayName("Tests de captura de piezas contra el trono.")
	class CapturaContraTrono {

		/**
		 * Comprueba que se captura un defensor contra el trono.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 * @version 1.0.0.1
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - -  - - 
		 * 4 R - - - D - - 
		 * 3 - - - - - A -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarCapturaDefensorContraTrono() throws CoordenadasIncorrectasException {
			// 1 atacante, 1 defensor y 1 rey
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 4, 5 }, { 3, 4 }, { 3, 0 } }, Color.NEGRO);
			arbitro.mover(fabricarJugada(tablero, 4, 5, 3, 5)); // atacante
			arbitro.realizarCapturasTrasMover(); // elimina al defensor

			assertAll("captura una pieza defensora en un solo movimiento contra trono",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(3, 4)).estaVacia(), is(true)), // Bug: corregido
					() -> assertThat("Deberían capturarse una pieza defensora.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(0)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(1)));
		}

		/**
		 * Comprueba que se captura un atacante contra el trono.
	     *
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 * @version 1.0.0.1
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - -  - - 
		 * 4 R - - - A - - 
		 * 3 - - - - - D -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarCapturaAtacanteContraTrono() throws CoordenadasIncorrectasException {
			// 1 atacante, 1 defensor y 1 rey
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 3, 4 }, { 4, 5 }, { 3, 0 } }, Color.BLANCO); // Bug: corregido valor númerico {3, 5} por {3,4}
			arbitro.mover(fabricarJugada(tablero, 4, 5, 3, 5)); // defensor
			arbitro.realizarCapturasTrasMover(); // elimina al atacante

			assertAll("captura una pieza atacante en un solo movimiento contra trono",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(3, 4)).estaVacia(), is(true)), // Bug: corregido
					() -> assertThat("Deberían capturarse una pieza atacante.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(0)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(1)));
		}
	}

	/**
	 * No se permiten suicidio de las piezas.
	 */
	@Nested
	@DisplayName("Tests que comprueban que las piezas no pueden suicidarse.")
	class SinSuicidios {

		/**
		 * Comprueba que no se captura un atacante que se coloca entre dos defensores.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - - 
		 * 4 D - D R - - - 
		 * 3 - A - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarNoHaySuicidioDelAtacanteAMitadDeTablero() throws CoordenadasIncorrectasException {
			// 1 atacante, 2 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 4, 1 }, { 3, 0 }, { 3, 2 }, { 3, 3 } }, Color.NEGRO);
			arbitro.mover(fabricarJugada(tablero, 4, 1, 3, 1)); // atacante se mueve entre dos defensores
			arbitro.realizarCapturasTrasMover(); // intentar eliminar piezas

			assertAll("no permite el suicidio de un atacante a mitad de tablero",
					() -> assertThat("No debería estar vacía la celda con atacante.",
							tablero.consultarCelda(new Coordenada(3, 1)).estaVacia(), is(false)),
					() -> assertThat("Debería estar el mismo número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(1)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(2)));
		}

		/**
		 * Comprueba que no se captura un atacante que se coloca entre dos defensores
		 * tras un movimiento del defensor.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - - 
		 * 4 D - D R - - D 
		 * 3 - A - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarNoHaySuicidioDelAtacanteAMitadDeTableroYSigueSinQuitarseTrasMovimientoDefensor() throws CoordenadasIncorrectasException {
			// 1 atacante, 2 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR,
							TipoPieza.REY },
					new int[][] { { 4, 1 }, { 3, 0 }, { 3, 2 }, { 3, 6 }, { 3, 3 } }, Color.NEGRO);
			arbitro.mover(fabricarJugada(tablero, 4, 1, 3, 1)); // atacante se mueve entre dos defensores
			arbitro.realizarCapturasTrasMover(); // intentar eliminar piezas
			arbitro.mover(fabricarJugada(tablero, 3, 6, 2, 6)); // mover defensor sin afectar a capturas
			arbitro.realizarCapturasTrasMover(); // volvemos a revisar si hay piezas que eliminar

			assertAll("no elimina piezas que intentaron suicidarse en el movimiento previo",
					() -> assertThat("No debería estar vacía la celda con atacante.",
							tablero.consultarCelda(new Coordenada(3, 1)).estaVacia(), is(false)),
					() -> assertThat("Debería estar el mismo número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(1)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(3)));
		}

		/**
		 * Comprueba que no se captura un atacante que se coloca entre un defensor y una provincia.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - D - - - -  
		 * 6 - A - - - - - 
		 * 5 - - - - - - - 
		 * 4 - - - R - - - 
		 * 3 - - - - - - -
		 * 2 - - - - - - -
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarNoHaySuicidioDelAtacanteContraProvincia() throws CoordenadasIncorrectasException {
			// 1 atacante, 1 defensor y 1 rey
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 1, 1 }, { 0, 2 }, { 3, 3 } }, Color.NEGRO);
			arbitro.mover(fabricarJugada(tablero, 1, 1, 0, 1)); // atacante se mueve entre defensor y provincia
			arbitro.realizarCapturasTrasMover(); // intentar eliminar piezas

			assertAll("no permite el suicidio de un atacante contra la provincia",
					() -> assertThat("No debería estar vacía la celda con atacante.",
							tablero.consultarCelda(new Coordenada(0, 1)).estaVacia(), is(false)),
					() -> assertThat("Debería estar el mismo número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(1)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(1)));
		}

		/**
		 * Comprueba que no se captura un atacante que se coloca entre un defensor y una provincia
		 * tras movimiento del defensor.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - D - - - -  
		 * 6 - A - - - - D
		 * 5 - - - - - - - 
		 * 4 - - - R - - - 
		 * 3 - - - - - - -
		 * 2 - - - - - - -
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarNoHaySuicidioDelAtacanteContraProvinciaYSigueSinQuitarseTrasMovimientoDefensor() throws CoordenadasIncorrectasException {
			// 1 atacante, 2 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 1, 1 }, { 0, 2 }, { 1, 6 }, { 3, 3 } }, Color.NEGRO);
			arbitro.mover(fabricarJugada(tablero, 1, 1, 0, 1)); // atacante se mueve entre defensor y provincia
			arbitro.realizarCapturasTrasMover(); // intentar eliminar piezas
			arbitro.mover(fabricarJugada(tablero, 1, 6, 2, 6)); // mover defensor sin afectar a capturas
			arbitro.realizarCapturasTrasMover(); // volvemos a revisar si hay piezas que eliminar

			assertAll("no elimina piezas que intentaron suicidarse en el movimiento previo",
					() -> assertThat("No debería estar vacía la celda con atacante.",
							tablero.consultarCelda(new Coordenada(0, 1)).estaVacia(), is(false)),
					() -> assertThat("Debería estar el mismo número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(1)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(2)));
		}

		/**
		 * Comprueba que no se captura un defensor que se coloca entre un atacante y el trono.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - - - -  
		 * 6 - R - - - - - 
		 * 5 - - - - - - - 
		 * 4 - - - - - - - 
		 * 3 - - - - D - -
		 * 2 - - - A - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarNoHaySuicidioDelDefensorContraTrono() throws CoordenadasIncorrectasException {
			// 1 atacante, 1 defensor y 1 rey
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 5, 3 }, { 4, 4 }, { 1, 1 } }, Color.BLANCO);
			arbitro.mover(fabricarJugada(tablero, 4, 4, 4, 3)); // defensor se mueve entre trono y atacante
			arbitro.realizarCapturasTrasMover(); // intentar eliminar piezas

			assertAll("no permite el suicidio de un defensor contra el trono",
					() -> assertThat("No debería estar vacía la celda con defensor.",
							tablero.consultarCelda(new Coordenada(4, 3)).estaVacia(), is(false)),
					() -> assertThat("Debería estar el mismo número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(1)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(1)));
		}

		/**
		 * Comprueba que no se captura un defensor que se coloca entre un atacante y el trono
		 * tras movimiento del atacante.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - - - -  
		 * 6 - R - - - A - 
		 * 5 - - - - - - - 
		 * 4 - - - - - - - 
		 * 3 - - - - D - -
		 * 2 - - - A - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarNoHaySuicidioDelDefensorContraTronoYSigueSinQuitarseTrasMovimientoAtacante() throws CoordenadasIncorrectasException {
			// 1 atacante, 2 defensores y 1 rey
			// 1 atacante, 1 defensor y 1 rey
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 5, 3 }, {1, 5}, { 4, 4 }, { 1, 1 } }, Color.BLANCO);
			arbitro.mover(fabricarJugada(tablero, 4, 4, 4, 3)); // defensor se mueve entre trono y atacante
			arbitro.realizarCapturasTrasMover(); // intentar eliminar piezas
			arbitro.mover(fabricarJugada(tablero, 1, 5, 1, 6)); // mover atacante sin afectar a capturas
			arbitro.realizarCapturasTrasMover(); // volvemos a revisar si hay piezas que eliminar

			assertAll("no elimina piezas que intentaron suicidarse en el movimiento previo",
					() -> assertThat("No debería estar vacía la celda con defensor.",
							tablero.consultarCelda(new Coordenada(4, 3)).estaVacia(), is(false)),
					() -> assertThat("Debería estar el mismo número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(2)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(1)));
		}
	}
	
	/**
	 * No se captura nunca al rey. Dicha pieza nunca se retira del tablero.
	 */
	@Nested
	@DisplayName("Tests para comprobar que nunca se captura el rey aunque esté derrotado.")
	class SinCapturaDelRey {
		
		/**
		 * Comprueba que no se captura el rey que está capturado por pinza de atacantes.
		 * tras un movimiento del atacante.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 A R - - A - - - 
		 * 4 - - - - - - -
 		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarNoSeCapturaElReyAunqueEsteDerrotadoEnHorizontal() throws CoordenadasIncorrectasException {
			// 2 atacantes 0 defensores y 1 rey
			// given
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, 
							TipoPieza.REY },
					new int[][] { { 2, 0 }, { 2, 4 }, { 2, 1 } }, Color.NEGRO);
			// when
			arbitro.mover(fabricarJugada(tablero, 2, 4, 2, 2)); // movemos atacante para hacer pinza sobre el rey
			arbitro.realizarCapturasTrasMover(); // volvemos a revisar si hay piezas que eliminar
			// then
			assertAll("no debería eliminar al rey aunque esté derrotado",
					() -> assertThat("No debería estar vacía la celda del rey.",
							tablero.consultarCelda(new Coordenada(2, 1)).estaVacia(), is(false)),
					() -> assertThat("Debería estar el mismo número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(2)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(0)));
		}
		
		
		/**
		 * Comprueba que no se captura el rey que está capturado por pinza de atacantes.
		 * tras un movimiento del atacante.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - A - -  
		 * 6 - - - - R - - 
		 * 5 - - - - - - - 
		 * 4 - - - - A - -
 		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarNoSeCapturaElReyAunqueEsteDerrotadoEnVertical() throws CoordenadasIncorrectasException {
			// 2 atacantes 0 defensores y 1 rey
			// given
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, 
							TipoPieza.REY },
					new int[][] { { 0, 4 }, { 3, 4 }, { 1, 4 } }, Color.NEGRO);
			// when
			arbitro.mover(fabricarJugada(tablero, 3, 4, 2, 4)); // movemos atacante para hacer pinza sobre el rey
			arbitro.realizarCapturasTrasMover(); // volvemos a revisar si hay piezas que eliminar
			// then
			assertAll("no debería eliminar al rey aunque esté derrotado",
					() -> assertThat("No debería estar vacía la celda del rey.",
							tablero.consultarCelda(new Coordenada(1, 4)).estaVacia(), is(false)),
					() -> assertThat("Debería estar el mismo número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(2)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(0)));
		}
	}
	
	/**
	 * No se permiten suicidio de las piezas incluso tras varios movimientos.
	 */
	@Nested
	@DisplayName("Tests que comprueban que las piezas no pueden suicidarse incluso tras varios movimientos.")
	class SinSuicidiosTrasRepetidosMovimientos {

		/**
		 * Comprueba que no se captura un atacante que se coloca entre dos defensores
		 * incluso tras varios movimientos.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - D - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - - 
		 * 4 D - D R - - - 
		 * 3 - A - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarNoHaySuicidioDelAtacanteAMitadDeTableroTrasVarioMovimientos() throws CoordenadasIncorrectasException {
			// 1 atacante, 2 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 4, 1 }, {0, 2}, { 3, 0 }, { 3, 2 }, { 3, 3 } }, Color.NEGRO);
			arbitro.mover(fabricarJugada(tablero, 4, 1, 3, 1)); // atacante se mueve entre dos defensores
			arbitro.realizarCapturasTrasMover(); // intentar eliminar piezas
			arbitro.cambiarTurno(); // no debería influir
			arbitro.mover(fabricarJugada(tablero, 0, 2, 0, 3)); // mueve el defensor en la parte superior
			arbitro.realizarCapturasTrasMover(); // intentar eliminar piezas

			assertAll("no permite el suicidio de un atacante a mitad de tablero incluso tras movimiento del defensor",
					() -> assertThat("No debería estar vacía la celda con atacante.",
							tablero.consultarCelda(new Coordenada(3, 1)).estaVacia(), is(false)),
					() -> assertThat("Debería estar el mismo número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(1)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(3)));
		}
		
		/**
		 * Comprueba que no se captura atacante que se coloca entre dos defensores
		 * incluso tras varios movimientos. Tampoco otro atacante que ya estaba
		 * entre dos defensores.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 * @version 1.0.0.1
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - D - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - - 
		 * 4 D - D R - - - 
		 * 3 - A - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - D A D -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarNoHaySuicidioDeVariosAtacantesTrasVarioMovimientos() throws CoordenadasIncorrectasException {
			// 2 atacante, 5 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE,
					TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 4, 1 }, {6, 4}, {0, 2}, { 3, 0 }, 
						{ 3, 2 }, { 6, 5 }, {6, 3}, {3, 3} }, Color.NEGRO);
			arbitro.mover(fabricarJugada(tablero, 4, 1, 3, 1)); // atacante se mueve entre dos defensores
			arbitro.realizarCapturasTrasMover(); // intentar eliminar piezas
			arbitro.cambiarTurno(); // no debería influir
			arbitro.mover(fabricarJugada(tablero, 0, 2, 0, 3)); // mueve el defensor en la parte superior
			arbitro.realizarCapturasTrasMover(); // intentar eliminar piezas

			assertAll("no permite el suicidio de varios atacantes a mitad de tablero incluso tras varios movimientos",
					() -> assertThat("No debería estar vacía la celda con atacante.",
							tablero.consultarCelda(new Coordenada(3, 1)).estaVacia(), is(false)),
					() -> assertThat("No debería estar vacía la celda con atacante.",
							tablero.consultarCelda(new Coordenada(6, 4)).estaVacia(), is(false)),
					() -> assertThat("Debería estar el mismo número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(2)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(5))); 
		}
		
		/**
		 * Comprueba que no se captura un defensor que se coloca entre dos atacantes
		 * incluso tras varios movimientos.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - A - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - - 
		 * 4 - - - - R - - 
		 * 3 A - A - - - -
		 * 2 - D - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarNoHaySuicidioDelDefensorAMitadDeTableroTrasVarioMovimientos() throws CoordenadasIncorrectasException {
			// 1 atacante, 2 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, 
							TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { {0, 2}, { 4, 0 }, { 4, 2 }, { 5, 1 }, { 3, 4 } }, Color.NEGRO);
			
			arbitro.mover(fabricarJugada(tablero, 5, 1, 4, 1)); // defensor se mueve entre dos atacantes
			arbitro.realizarCapturasTrasMover(); // intentar eliminar piezas
			
			arbitro.mover(fabricarJugada(tablero, 0, 2, 0, 3)); // mueve el atacante en la parte superior
			arbitro.realizarCapturasTrasMover(); // intentar eliminar piezas
			
			assertAll("no permite el suicidio de un atacante a mitad de tablero incluso tras movimiento del defensor",
					() -> assertThat("No debería estar vacía la celda con defensor.",
							tablero.consultarCelda(new Coordenada(4, 1)).estaVacia(), is(false)),
					() -> assertThat("Debería estar el mismo número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(3)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(1)));
		}
		
		/**
		 * Comprueba que no se captura defensor que se coloca entre dos atacantes
		 * incluso tras varios movimientos. Tampoco otro defensor que ya estaba
		 * entre dos atacantes.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - A - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - - 
		 * 4 - - - - R - - 
		 * 3 A - A - - - -
		 * 2 - D - - - - - 
		 * 1 - - - A D A -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarNoHaySuicidioDeVariosDefensoresTrasVarioMovimientos() throws CoordenadasIncorrectasException {
			// 2 atacante, 5 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE,
					TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { {0, 2}, { 4, 0 }, { 4, 2 }, {6, 3}, {6, 5},
						{ 5, 1 }, {6, 4}, { 3, 4 } }, Color.NEGRO);
			arbitro.mover(fabricarJugada(tablero, 5, 1, 4, 1)); // defensor se mueve entre dos atacantes
			arbitro.realizarCapturasTrasMover(); // intentar eliminar piezas
			arbitro.cambiarTurno(); // no debería influir
			arbitro.mover(fabricarJugada(tablero, 0, 2, 0, 3)); // mueve el defensor en la parte superior
			arbitro.realizarCapturasTrasMover(); // intentar eliminar piezas

			assertAll("no permite el suicidio de varios defensores a mitad de tablero incluso tras varios movimientos",
					() -> assertThat("No debería estar vacía la celda con defensor.",
							tablero.consultarCelda(new Coordenada(4, 1)).estaVacia(), is(false)),
					() -> assertThat("No debería estar vacía la celda con defensor.",
							tablero.consultarCelda(new Coordenada(6, 4)).estaVacia(), is(false)),
					() -> assertThat("Debería estar el mismo número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(5)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(2)));
		}
		
		/**
		 * Comprueba que no se captura al rey que se coloca entre dos atacantes
		 * incluso tras varios movimientos.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - A - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - - 
		 * 4 - - - - - - - 
		 * 3 A - A - - - -
		 * 2 - R - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarNoHaySuicidioDelReyAMitadDeTableroTrasVarioMovimientos() throws CoordenadasIncorrectasException {
			// 1 atacante, 2 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, 
							TipoPieza.REY },
					new int[][] { {0, 2}, { 4, 0 }, { 4, 2 }, { 5, 1 }}, Color.NEGRO);
			
			arbitro.mover(fabricarJugada(tablero, 5, 1, 4, 1)); // rey se mueve entre dos atacantes
			arbitro.realizarCapturasTrasMover(); // intentar eliminar piezas
			
			arbitro.mover(fabricarJugada(tablero, 0, 2, 0, 3)); // mueve el atacante en la parte superior
			arbitro.realizarCapturasTrasMover(); // intentar eliminar piezas
			
			assertAll("no permite el suicidio del rey a mitad de tablero incluso tras movimiento del atacante",
					() -> assertThat("No debería haber victoria del atacante.",
							arbitro.haGanadoAtacante(), is(false)),
					() -> assertThat("No debería estar vacía la celda con rey.",
							tablero.consultarCelda(new Coordenada(4, 1)).estaVacia(), is(false)),
					() -> assertThat("Debería estar el mismo número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(3)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(0)));
		}
		
		
		/**
		 * Comprueba que no se captura rey que se coloca entre dos atacantes
		 * incluso tras varios movimientos. Tampoco otro defensor que ya estaba
		 * entre dos atacantes.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - A - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - - 
		 * 4 - - - - - - - 
		 * 3 A - A - - - -
		 * 2 - R - - - - - 
		 * 1 - - - A D A -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		void comprobarNoHaySuicidioDeReyYDefensorTrasVarioMovimientos() throws CoordenadasIncorrectasException {
			// 2 atacante, 5 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE,
					TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { {0, 2}, { 4, 0 }, { 4, 2 }, {6, 3}, {6, 5},
					 {6, 4}, { 5, 1 } }, Color.NEGRO);
			arbitro.mover(fabricarJugada(tablero, 5, 1, 4, 1)); // rey se mueve entre dos atacantes
			arbitro.realizarCapturasTrasMover(); // intentar eliminar piezas
			arbitro.cambiarTurno(); // no debería influir
			arbitro.mover(fabricarJugada(tablero, 0, 2, 0, 3)); // mueve el atacante en la parte superior
			arbitro.realizarCapturasTrasMover(); // intentar eliminar piezas

			assertAll("no permite el suicidio de rey y defensor a mitad de tablero incluso tras varios movimientos",
					() -> assertThat("No debería haber victoria del atacante.",
							arbitro.haGanadoAtacante(), is(false)),
					() -> assertThat("No debería estar vacía la celda con rey.",
							tablero.consultarCelda(new Coordenada(4, 1)).estaVacia(), is(false)),
					() -> assertThat("No debería estar vacía la celda con defensor.",
							tablero.consultarCelda(new Coordenada(6, 5)).estaVacia(), is(false)),
					() -> assertThat("Debería estar el mismo número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(5)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(1)));
		}
	}

}
