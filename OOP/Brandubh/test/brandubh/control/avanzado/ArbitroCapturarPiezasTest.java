package brandubh.control.avanzado;

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
import brandubh.util.Coordenada;
import brandubh.util.TipoPieza;


/**
 * Comprobación de captura de piezas en distintas condiciones.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@DisplayName("Tests del Arbitro sobre capturas de piezas.")
@Timeout(value = 2, unit = TimeUnit.SECONDS) // Time out global para todos los tests salvo los de ciclo de vida
public class ArbitroCapturarPiezasTest {

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
	 * Comprueba las condiciones de no victoria cuando el rey no está rodeado usando
	 * el trono.
	 */
	private void comprobarNoVictoriaDelAtacante() {
		assertAll("no victoria rodeando al rey contra trono",
				() -> assertThat("No debería ganar el atacante.", arbitro.haGanadoAtacante(), is(false)),
				() -> assertThat("No debería ganar el defensor.", arbitro.haGanadoRey(), is(false)));
	}	

	/**
	 * Captura de atacante contra provincia moviendo un defensor.
	 */
	@Nested
	@DisplayName("Tests de capturas de atacante contra provincia moviendo un defensor.")
	class CapturaAtacanteContraProvincia {

		/**
		 * Inicializa con todas las piezas colocadas.
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - A - - -  
		 * 6 - - - A - - - 
		 * 5 - - - D - - - 
		 * 4 A A D R D A A 
		 * 3 - - - D - - -
		 * 2 - - - A - - - 
		 * 1 - - - A - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		@BeforeEach
		void inicializar() {
			arbitro.colocarPiezasConfiguracionInicial();
		}

		/**
		 * Comprueba la captura de atacante contra provincia en esquina inferior 
		 * derecha en horizontal.
		 */
		@Test
		@DisplayName("Comprueba la captura de atacante contra provincia en esquina inferior derecha en horizontal.")
		void comprobarCapturaAtacanteContraProvinciaEnEsquinaInferiorDerechaEnHorizontal() {
			arbitro.mover(fabricarJugada(tablero, 6, 3, 6, 5)); // atacante
			arbitro.mover(fabricarJugada(tablero, 3, 4, 6, 4)); // defensor
			arbitro.realizarCapturasTrasMover();
			assertAll("captura de pieza atacante",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(6, 5)).estaVacia(), is(true)),
					() -> assertThat("No debería capturarse una pieza defensora.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
					() -> assertThat("Debería capturarse una pieza atacante.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(7)));
		}

		/**
		 * Comprueba la captura de atacante contra provincia en esquina superior
		 * derecha en horizontal.
		 */
		@Test
		@DisplayName("Comprueba la captura de atacante contra provincia en esquina superior derecha en horizontal.")
		void comprobarCapturaAtacanteContraProvinciaEnEsquinaSuperiorDerechaEnHorizontal() {
			arbitro.mover(fabricarJugada(tablero, 3, 5, 0, 5)); // atacante
			arbitro.mover(fabricarJugada(tablero, 3, 4, 0, 4)); // defensor
			arbitro.realizarCapturasTrasMover();
			assertAll("captura de pieza atacante",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(0, 5)).estaVacia(), is(true)),
					() -> assertThat("No debería capturarse una pieza defensora.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
					() -> assertThat("Debería capturarse una pieza atacante.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(7)));
		}

		/**
		 * Comprueba la captura de atacante contra provincia en esquina inferior 
		 * izquierda en horizontal.
		 */
		@Test
		@DisplayName("Comprueba la captura de atacante contra provincia en esquina inferior izquierda en horizontal.")
		void comprobarCapturaAtacanteContraProvinciaEnEsquinaInferiorIzquierdaEnHorizontal() {
			arbitro.mover(fabricarJugada(tablero, 3, 1, 6, 1)); // atacante
			arbitro.mover(fabricarJugada(tablero, 3, 2, 6, 2)); // defensor
			arbitro.realizarCapturasTrasMover();
			assertAll("captura de pieza atacante",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(6, 1)).estaVacia(), is(true)),
					() -> assertThat("No debería capturarse una pieza defensora.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
					() -> assertThat("Debería capturarse una pieza atacante.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(7)));
		}

		/**
		 * Comprueba la captura de atacante contra provincia en esquina superior
		 * izquierda en horizontal.
		 */
		@Test
		@DisplayName("Comprueba la captura de atacante contra provincia en esquina superior izauierda en horizontal.")
		void comprobarCapturaAtacanteContraProvinciaEnEsquinaSuperiorIzquierdaEnHorizontal() {
			arbitro.mover(fabricarJugada(tablero, 3, 1, 0, 1)); // atacante
			arbitro.mover(fabricarJugada(tablero, 3, 2, 0, 2)); // defensor
			arbitro.realizarCapturasTrasMover();
			assertAll("captura de pieza atacante",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(0, 1)).estaVacia(), is(true)),
					() -> assertThat("No debería capturarse una pieza defensora.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
					() -> assertThat("Debería capturarse una pieza atacante.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(7)));
		}

		/**
		 * Comprueba la captura de atacante contra provincia en esquina superior 
		 * izquierda en vertical.
		 */
		@Test
		@DisplayName("Comprueba la captura de atacante contra provincia en esquina superior izquierda en vertical.")
		void comprobarCapturaAtacanteContraProvinciaEnEsquinaSuperiorIzquierdaEnVertical() {
			arbitro.mover(fabricarJugada(tablero, 1, 3, 1, 0)); // atacante
			arbitro.mover(fabricarJugada(tablero, 2, 3, 2, 0)); // defensor
			arbitro.realizarCapturasTrasMover();
			assertAll("captura de pieza atacante",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(1, 0)).estaVacia(), is(true)),
					() -> assertThat("No debería capturarse una pieza defensora.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
					() -> assertThat("Debería capturarse una pieza atacante.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(7)));
		}

		/**
		 * Comprueba la captura de atacante contra provincia en esquina superior
		 * derecha en vertical.
		 */
		@Test		
		@DisplayName("Comprueba la captura de atacante contra provincia en esquina superior derecha en vertical.")
		void comprobarCapturaAtacanteContraProvinciaEnEsquinaSuperiorDerechaEnVertical() {
			arbitro.mover(fabricarJugada(tablero, 1, 3, 1, 6)); // atacante
			arbitro.mover(fabricarJugada(tablero, 2, 3, 2, 6)); // defensor
			arbitro.realizarCapturasTrasMover();
			assertAll("captura de pieza atacante",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(1, 6)).estaVacia(), is(true)),
					() -> assertThat("No debería capturarse una pieza defensora.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
					() -> assertThat("Debería capturarse una pieza atacante.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(7)));
		}

		/**
		 * Comprueba la captura de atacante contra provincia en esquina inferior 
		 * izquierda en vertical.
		 */
		@Test
		@DisplayName("Comprueba la captura de atacante contra provincia en esquina inferior izquierda en vertical.")
		void comprobarCapturaAtacanteContraProvinciaEnEsquinaInferiorIzquierdaEnVertical() {
			arbitro.mover(fabricarJugada(tablero, 5, 3, 5, 0)); // atacante
			arbitro.mover(fabricarJugada(tablero, 4, 3, 4, 0)); // defensor
			arbitro.realizarCapturasTrasMover();
			assertAll("captura de pieza atacante",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(5, 0)).estaVacia(), is(true)),
					() -> assertThat("No debería capturarse una pieza defensora.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
					() -> assertThat("Debería capturarse una pieza atacante.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(7)));
		}

		/**
		 * Comprueba la captura de atacante contra provincia en esquina inferior 
		 * derecha en vertical.
		 */
		@Test
		@DisplayName("Comprueba la captura de atacante contra provincia en esquina inferior derecha en vertical.")
		void comprobarCapturaAtacanteContraProvinciaEnEsquinaInferiorDerechaEnVertical() {
			arbitro.mover(fabricarJugada(tablero, 5, 3, 5, 6)); // atacante
			arbitro.mover(fabricarJugada(tablero, 4, 3, 4, 6)); // defensor
			arbitro.realizarCapturasTrasMover();
			assertAll("captura de pieza atacante",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(5, 6)).estaVacia(), is(true)),
					() -> assertThat("No debería capturarse una pieza defensora.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
					() -> assertThat("Debería capturarse una pieza atacante.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(7)));
		}
	}

	/**
	 * Captura de defensor con dos piezas atacantes.
	 */
	@Nested
	@DisplayName("Tests de captura de defensor con dos piezas atacantes.")
	class CapturaDefensorEntreMediasDeDosAtacantes {

		/**
		 * Inicializa con todas las piezas colocadas.
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - A - - -  
		 * 6 - - - A - - - 
		 * 5 - - - D - - - 
		 * 4 A A D R D A A 
		 * 3 - - - D - - -
		 * 2 - - - A - - - 
		 * 1 - - - A - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		@BeforeEach
		void inicializar() {
			arbitro.colocarPiezasConfiguracionInicial();
		}

		/**
		 * Comprueba la captura de defensor en cuadrante izquierdo superior en vertical.
		 */
		@Test
		@DisplayName("Comprueba la captura de defensor en cuadrante izquierdo superior en vertical.")
		void comprobarCapturaDefensorEnCuadranteIzquierdoSuperiorEnVertical() {
			arbitro.mover(fabricarJugada(tablero, 0, 3, 0, 1)); // atacante
			arbitro.mover(fabricarJugada(tablero, 2, 3, 2, 1)); // defensor
			arbitro.mover(fabricarJugada(tablero, 0, 1, 1, 1)); // atacante
			arbitro.realizarCapturasTrasMover();

			assertAll("captura de pieza del defensor",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(2, 1)).estaVacia(), is(true)),
					() -> assertThat("Deberían estar el número inicial de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
					() -> assertThat("Debería capturarse una pieza del defensor.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(3)));
		}

		/**
		 * Comprueba la captura de defensor en cuadrante derecho superior en vertical.
		 */
		@Test
		@DisplayName("Comprueba la captura de defensor en cuadrante derecho superior en vertical.")
		void comprobarCapturaDefensorEnCuadranteDerechoSuperiorEnVertical() {
			arbitro.mover(fabricarJugada(tablero, 0, 3, 0, 5)); // atacante
			arbitro.mover(fabricarJugada(tablero, 2, 3, 2, 5)); // defensor
			arbitro.mover(fabricarJugada(tablero, 0, 5, 1, 5)); // atacante
			arbitro.realizarCapturasTrasMover();

			assertAll("captura de pieza del defensor",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(2, 5)).estaVacia(), is(true)),
					() -> assertThat("Deberían estar el número inicial de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
					() -> assertThat("Debería capturarse una pieza del defensor.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(3)));
		}

		/**
		 * Comprueba la captura de defensor en cuadrante izquierdo inferior en vertical.
		 */
		@Test
		@DisplayName("Comprueba la captura de defensor en cuadrante izquierdo inferior en vertical.")
		void comprobarCapturaDefensorEnCuadranteIzquierdoInferiorEnVertical() {
			arbitro.mover(fabricarJugada(tablero, 6, 3, 6, 1)); // atacante
			arbitro.mover(fabricarJugada(tablero, 4, 3, 4, 1)); // defensor
			arbitro.mover(fabricarJugada(tablero, 6, 1, 5, 1)); // atacante
			arbitro.realizarCapturasTrasMover();

			assertAll("captura de pieza del defensor",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(4, 1)).estaVacia(), is(true)),
					() -> assertThat("Deberían estar el número inicial de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
					() -> assertThat("Debería capturarse una pieza del defensor.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(3)));
		}

		/**
		 * Comprueba la captura de defensor en cuadrante derecho inferior en vertical.
		 */
		@Test
		@DisplayName("Comprueba la captura de defensor en cuadrante derecho inferior en vertical.")
		void comprobarCapturaDefensorEnCuadranteDerechoInferiorEnVertical() {
			arbitro.mover(fabricarJugada(tablero, 6, 3, 6, 5)); // atacante
			arbitro.mover(fabricarJugada(tablero, 4, 3, 4, 5)); // defensor
			arbitro.mover(fabricarJugada(tablero, 6, 5, 5, 5)); // atacante
			arbitro.realizarCapturasTrasMover();

			assertAll("captura de pieza del defensor",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(4, 5)).estaVacia(), is(true)),
					() -> assertThat("Deberían estar el número inicial de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
					() -> assertThat("Debería capturarse una pieza del defensor.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(3)));
		}

		/**
		 * Comprueba la captura de defensor en cuadrante izquierdo superior en horizontal.
		 */
		@Test
		@DisplayName("Comprueba la captura de defensor en cuadrante izquierdo superior en horizontal.")
		void comprobarCapturaDefensorEnCuadranteIzquierdoSuperiorEnHorizontal() {
			arbitro.mover(fabricarJugada(tablero, 3, 0, 1, 0)); // atacante
			arbitro.mover(fabricarJugada(tablero, 3, 2, 1, 2)); // defensor
			arbitro.mover(fabricarJugada(tablero, 1, 0, 1, 1)); // atacante
			arbitro.realizarCapturasTrasMover();

			assertAll("captura de pieza del defensor",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(1, 2)).estaVacia(), is(true)),
					() -> assertThat("Deberían estar el número inicial de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
					() -> assertThat("Debería capturarse una pieza del defensor.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(3)));
		}

		/**
		 * Comprueba la captura de defensor en cuadrante derecho superior en horizontal.
		 */
		@Test
		@DisplayName("Comprueba la captura de defensor en cuadrante derecho superior en horizontal.")
		void comprobarCapturaDefensorEnCuadranteDerechoSuperiorEnHorizontal() {
			arbitro.mover(fabricarJugada(tablero, 3, 6, 1, 6)); // atacante
			arbitro.mover(fabricarJugada(tablero, 3, 4, 1, 4)); // defensor
			arbitro.mover(fabricarJugada(tablero, 1, 6, 1, 5)); // atacante
			arbitro.realizarCapturasTrasMover();

			assertAll("captura de pieza del defensor",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(1, 4)).estaVacia(), is(true)),
					() -> assertThat("Deberían estar el número inicial de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
					() -> assertThat("Debería capturarse una pieza del defensor.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(3)));
		}

		/**
		 * Comprueba la captura de defensor en cuadrante izquierdo inferior en horizontal.
		 */
		@Test
		@DisplayName("Comprueba la captura de defensor en cuadrante izquierdo inferior en horizontal.")
		void comprobarCapturaDefensorEnCuadranteIzquierdoInferiorEnHorizontal() {
			arbitro.mover(fabricarJugada(tablero, 3, 0, 5, 0)); // atacante
			arbitro.mover(fabricarJugada(tablero, 3, 2, 5, 2)); // defensor
			arbitro.mover(fabricarJugada(tablero, 5, 0, 5, 1)); // atacante
			arbitro.realizarCapturasTrasMover();

			assertAll("captura de pieza del defensor",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(5, 2)).estaVacia(), is(true)),
					() -> assertThat("Deberían estar el número inicial de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
					() -> assertThat("Debería capturarse una pieza del defensor.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(3)));
		}

		/**
		 * Comprueba la captura de defensor en cuadrante derecho inferior en horizontal.
		 */
		@Test
		@DisplayName("Comprueba la captura de defensor en cuadrante derecho inferior en horizontal.")
		void comprobarCapturaDefensorEnCuadranteDerechoInferiorEnHorizontal() {
			arbitro.mover(fabricarJugada(tablero, 3, 6, 5, 6)); // atacante
			arbitro.mover(fabricarJugada(tablero, 3, 4, 5, 4)); // defensor
			arbitro.mover(fabricarJugada(tablero, 5, 6, 5, 5)); // atacante
			arbitro.realizarCapturasTrasMover();

			assertAll("captura de pieza del defensor",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(5, 4)).estaVacia(), is(true)),
					() -> assertThat("Deberían estar el número inicial de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
					() -> assertThat("Debería capturarse una pieza del defensor.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(3)));
		}
	}

	/**
	 * Captura de tres piezas en una sola jugada.
	 */
	@Nested
	@DisplayName("Tests de captura de tres piezas en una sola jugada.")
	class CapturarTresPiezasEnUnaSolaJugada {

		/**
		 * Comprueba que se capturan tres defensores en una sola jugada.
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - A - - - - 
		 * 4 - - - R - - - 
		 * 3 A D - D A - -
		 * 2 - - D - - - - 
		 * 1 - - A - - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		@Test
		@DisplayName("Comprueba la captura de tres piezas defensoras en una sola jugada.")
		void comprobarCapturaTresPiezasDefensorasEnUnaSolaJugada() {
			// 4 atacantes, 3 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE,
							TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 4, 0 }, { 2, 2 }, { 4, 4 }, { 6, 2 }, { 4, 1 }, { 4, 3 }, { 5, 2 }, { 3, 3 } },
					Color.NEGRO);
			arbitro.mover(fabricarJugada(tablero, 2, 2, 4, 2)); // atacante
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
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - D - - 
		 * 4 - - - R A - - 
		 * 3 - - D - - A D
		 * 2 - - - - A - - 
		 * 1 - - - - D - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		@Test
		@DisplayName("Comprueba la captura de tres piezas atacantes en una sola jugada.")
		void comprobarCapturaTresPiezasAtacantesEnUnaSolaJugada() {
			// 4 atacantes, 3 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.DEFENSOR,
							TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 3, 4 }, { 4, 5 }, { 5, 4 }, { 2, 4 }, { 4, 2 }, { 4, 6 }, { 6, 4 }, { 3, 3 } },
					Color.BLANCO);
			arbitro.mover(fabricarJugada(tablero, 4, 2, 4, 4)); // defensor
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
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - - - -  
		 * 6 D - - - - - - 
		 * 5 A - - - - - - 
		 * 4 - - D R - - - 
		 * 3 A - - - - - -
		 * 2 D - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		@DisplayName("Comprueba la captura de dos piezas atacantes en una sola jugada.")
		void comprobarCapturaDosPiezasAtacantesEnUnaSolaJugada() {
			// 2 atacantes, 3 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR,
							TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 2, 0 }, { 4, 0 }, { 1, 0 }, { 3, 2 }, { 5, 0 }, { 3, 3 } }, Color.BLANCO);
			arbitro.mover(fabricarJugada(tablero, 3, 2, 3, 0)); // defensor
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
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - A D - D A -  
		 * 6 - - - - - - - 
		 * 5 - - - A - - - 
		 * 4 - - - R - - - 
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		@DisplayName("Comprueba la captura de dos piezas defensoras en una sola jugada.")
		void comprobarCapturaDosPiezasDefensorasEnUnaSolaJugada() {
			// 3 atacantes, 2 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.DEFENSOR,
							TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 0, 1 }, { 0, 5 }, { 2, 3 }, { 0, 2 }, { 0, 4 }, { 3, 3 } }, Color.NEGRO);
			arbitro.mover(fabricarJugada(tablero, 2, 3, 0, 3)); // atacante
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
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - A 
		 * 5 - - - - D - - 
		 * 4 - - - R - - A 
		 * 3 - - - - - - D
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		@DisplayName("Comprueba la captura de dos atacantes una sola jugada contra provincia.")
		void comprobarCapturaDosPiezasAtacantesEnUnSoloMovimientoContraProvincia() {
			// 2 atacantes, 2 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR,
							TipoPieza.REY },
					new int[][] { { 1, 6 }, { 3, 6 }, { 2, 4 }, { 4, 6 }, { 3, 3 } }, Color.BLANCO);
			arbitro.mover(fabricarJugada(tablero, 2, 4, 2, 6)); // defensor
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
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - -  - - 
		 * 4 - - - R - - - 
		 * 3 - - - - A - -
		 * 2 - - - - - - - 
		 * 1 - - A D - D -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		@DisplayName("Comprueba la captura de dos defensores una sola jugada contra provincia.")
		void comprobarCapturaDosPiezasDefensorasEnUnSoloMovimientoContraProvincia() {
			// 2 atacantes, 2 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR,
							TipoPieza.REY },
					new int[][] { { 4, 4 }, { 6, 2 }, { 6, 3 }, { 6, 5 }, { 3, 3 } }, Color.BLANCO);
			arbitro.mover(fabricarJugada(tablero, 4, 4, 6, 4)); // atacante
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
		@DisplayName("Comprueba la captura de un defensor contra el trono.")
		void comprobarCapturaDefensorContraTrono() {
			// 1 atacante, 1 defensor y 1 rey
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 4, 5 }, { 3, 4 }, { 3, 0 } }, Color.NEGRO);
			arbitro.mover(fabricarJugada(tablero, 4, 5, 3, 5)); // atacante
			arbitro.realizarCapturasTrasMover(); // elimina al defensor

			assertAll("captura una pieza defensora en un solo movimiento contra trono",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(4, 3)).estaVacia(), is(true)),
					() -> assertThat("Deberían capturarse una pieza defensora.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(0)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(1)));
		}

		/**
		 * Comprueba que se captura un atacante contra el trono.
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
		@DisplayName("Comprueba la captura de un atacante contra el trono.")
		void comprobarCapturaAtacanteContraTrono() {
			// 1 atacante, 1 defensor y 1 rey
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 3, 5 }, { 4, 5 }, { 3, 0 } }, Color.BLANCO);
			arbitro.mover(fabricarJugada(tablero, 4, 5, 3, 5)); // defensor
			arbitro.realizarCapturasTrasMover(); // elimina al atacante

			assertAll("captura una pieza atacante en un solo movimiento contra trono",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(4, 3)).estaVacia(), is(true)),
					() -> assertThat("Deberían capturarse una pieza atacante.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(0)),
					() -> assertThat("Debería seguir estando el rey.", tablero.consultarNumeroPiezas(TipoPieza.REY),
							is(1)),
					() -> assertThat("Debería estar el mismo número de piezas defensoras.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(1)));
		}
	}

	/**
	 * No se permiten suicidio de las piezas atacantes y defensoras.
	 */
	@Nested
	@DisplayName("Tests que comprueban que las piezas no pueden suicidarse.")
	class SinSuicidios {

		/**
		 * Comprueba que no se captura un atacante que se coloca entre dos defensores.
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
		@DisplayName("Comprueba que no hay suicidio de un atacante a mitad de tablero.")
		void comprobarNoHaySuicidioDelAtacanteAMitadDeTablero() {
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
		@DisplayName("Comprueba que no hay suicidio de un atacante a mitad de tablero y sigue sin eliminar pieza tras movimiento del defensor.")
		void comprobarNoHaySuicidioDelAtacanteAMitadDeTableroYSigueSinQuitarseTrasMovimientoDefensor() {
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
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - D - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - - 
		 * 4 - - - R - - - 
		 * 3 - A - - - - -
		 * 2 - - - - - - -
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		@DisplayName("Comprueba que no hay suicidio de un atacante contra provincia.")
		void comprobarNoHaySuicidioDelAtacanteContraProvincia() {
			// 1 atacante, 1 defensor y 1 rey
			arbitro.colocarPiezas(new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 4, 1 }, { 0, 2 }, { 3, 3 } }, Color.NEGRO);
			arbitro.mover(fabricarJugada(tablero, 4, 1, 0, 1)); // atacante se mueve entre defensor y provincia
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
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - D - - - -  
		 * 6 - - - - - - D
		 * 5 - - - - - - - 
		 * 4 - - - R - - - 
		 * 3 - A - - - - -
		 * 2 - - - - - - -
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		@DisplayName("Comprueba que no hay suicidio de un atacante contrar provincia y sigue sin eliminar pieza tras movimiento del defensor.")
		void comprobarNoHaySuicidioDelAtacanteContraProvinciaYSigueSinQuitarseTrasMovimientoDefensor() {
			// 1 atacante, 2 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 4, 1 }, { 0, 2 }, { 1, 6 }, { 3, 3 } }, Color.NEGRO);
			arbitro.mover(fabricarJugada(tablero, 4, 1, 0, 1)); // atacante se mueve entre defensor y provincia
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
		@DisplayName("Comprueba que no hay suicidio del defensor contra el trono.")
		void comprobarNoHaySuicidioDelDefensorContraTrono() {
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
		@DisplayName("Comprueba que no hay suicidio del defensor contra el trono y sigue sin eliminar pieza tras movimiento del atacante.")
		void comprobarNoHaySuicidioDelDefensorContraTronoYSigueSinQuitarseTrasMovimientoAtacante() {
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
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - - 
		 * 4 A R - - A - -
 		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		@DisplayName("Comprueba que no se captura el rey aunque esté derrotado en horizontal.")
		void comprobarNoSeCapturaElReyAunqueEsteDerrotadoEnHorizontal() {
			// 2 atacantes 0 defensores y 1 rey
			// given
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, 
							TipoPieza.REY },
					new int[][] { { 3, 0 }, { 3, 4 }, { 3, 1 } }, Color.NEGRO);
			// when
			arbitro.mover(fabricarJugada(tablero, 3, 4, 3, 2)); // movemos atacante para hacer pinza sobre el rey
			arbitro.realizarCapturasTrasMover(); // volvemos a revisar si hay piezas que eliminar
			// then
			assertAll("no debería eliminar al rey aunque esté derrotado",
					() -> assertThat("No debería estar vacía la celda del rey.",
							tablero.consultarCelda(new Coordenada(3, 1)).estaVacia(), is(false)),
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
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - - A - -  
		 * 6 - - - - R - - 
		 * 5 - - - - - - - 
		 * 4 - - - - - - -
 		 * 3 - - - - A - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		// @formatter:on
		@Test
		@DisplayName("Comprueba que no se captura el rey aunque esté derrotado en vertical.")
		void comprobarNoSeCapturaElReyAunqueEsteDerrotadoEnVertical() {
			// 2 atacantes 0 defensores y 1 rey
			// given
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, 
							TipoPieza.REY },
					new int[][] { { 0, 4 }, { 4, 4 }, { 1, 4 } }, Color.NEGRO);
			// when
			arbitro.mover(fabricarJugada(tablero, 4, 4, 2, 4)); // movemos atacante para hacer pinza sobre el rey
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
		@DisplayName("Comprueba que no se permite el suicidio de un atacante tras varios movimientos.")
		void comprobarNoHaySuicidioDelAtacanteAMitadDeTableroTrasVarioMovimientos() {
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
		@DisplayName("Comprueba que no se permite el suicidio de varios atacantes tras varios movimientos.")
		void comprobarNoHaySuicidioDeVariosAtacantesTrasVarioMovimientos() {
			// 2 atacante, 5 defensores y 1 rey
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE,
					TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.REY },
					new int[][] { { 4, 1 }, {6, 4}, {0, 2}, { 3, 0 }, 
						{ 3, 2 }, { 3, 3 }, {6, 3}, {6, 5} }, Color.NEGRO);
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
		@DisplayName("Comprueba que no se permite el suicidio de un defensor a mitad del tablero tras varios movimientos.")
		void comprobarNoHaySuicidioDelDefensorAMitadDeTableroTrasVarioMovimientos() {
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
		@DisplayName("Comprueba que no se permite el suicidio de varios defensores tras varios movimientos.")
		void comprobarNoHaySuicidioDeVariosDefensoresTrasVarioMovimientos() {
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
		@DisplayName("Comprueba que no se permite el suicidio del rey a mitad del tablero tras varios movimientos.")
		void comprobarNoHaySuicidioDelReyAMitadDeTableroTrasVarioMovimientos() {
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
		@DisplayName("Comprueba que no se permite el suicidio del rey y defensor tras varios movimientos.")
		void comprobarNoHaySuicidioDeReyYDefensorTrasVarioMovimientos() {
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

	/**
	 * No victoria del atacante si es el rey el que se mueve entre tres piezas del atacante
	 * con el trono colindante. 
	 * 
	 * Un rey no puede provocar su propia captura directamente.
	 */
	@Nested
	@DisplayName("Tests para comprobar que no hay victoria del atacante si es el rey el que se mete entre tres piezas del atacante con el trono colindante.")
	class SinSuicicioDelReySiMueveEntreTresPiezasDelAtacanteConElTronoColindante {
		
		/**
		 * Comprueba no hay victoria rodeando si mueve el rey entrando por el norte.
		 */
		@Test
		@DisplayName("No victoria si es el rey el que mueve por el norte.")
		void comprobarNoVictoriaEntrandoElReyPorNorte() {
			entrarElReyPorElNorte();
			comprobarNoVictoriaDelAtacante();
		}
		
		/**
		 * Comprueba no hay victoria rodeando si mueve el rey entrando por el sur.
		 */
		@Test
		@DisplayName("No victoria si es el rey el que mueve por el sur.")
		void comprobarNoVictoriaEntrandoElReyPorSur() {
			entrarElReyPorElSur();
			comprobarNoVictoriaDelAtacante();
		}
		
		/**
		 * Comprueba no hay victoria rodeando si mueve el rey entrando por el este.
		 */
		@Test
		@DisplayName("No victoria si es el rey el que mueve por el este.")
		void comprobarNoVictoriaEntrandoElReyPorEste() {
			entrarElReyPorElEste();
			comprobarNoVictoriaDelAtacante();
		}
		
		/**
		 * Comprueba no hay victoria rodeando si mueve el rey entrando por el oeste.
		 */
		@Test
		@DisplayName("No victoria si es el rey el que mueve por el oeste.")
		void comprobarNoVictoriaEntrandoElReyPorOeste() {
			entrarElReyPorElOeste();
			comprobarNoVictoriaDelAtacante();
		}
		
		/**
		 * Coloca pieza del rey entrando por el norte.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - - 
		 * 4 - - - R - - -
		 * 3 - - A - A - -
		 * 2 - - - A - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		void entrarElReyPorElNorte() {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, 
							TipoPieza.REY },
					new int[][] { { 4, 2 }, { 4, 4 }, { 5, 3}, { 3, 3 } }, Color.BLANCO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 3, 4, 3));
		}
		

		/**
		 * Coloca pieza del rey entrando por el sur..
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - A - - - 
		 * 5 - - A - A - - 
		 * 4 - - - R - - -
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		void entrarElReyPorElSur() {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, 
							TipoPieza.REY },
					new int[][] { { 2, 2 }, { 2, 4 }, { 1, 3}, { 3, 3 } }, Color.BLANCO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 3, 2, 3));
		}
		
		/**
		 * Coloca pieza del rey entrando por el este.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - A - - - - 
		 * 4 - A - R - - -
		 * 3 - - A - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		void entrarElReyPorElEste() {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, 
							TipoPieza.REY },
					new int[][] { { 2, 2 }, { 3, 1 }, { 4, 2}, { 3, 3 } }, Color.BLANCO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 3, 3, 2));
		}
		
		/**
		 * Coloca pieza del rey entrando por el oeste.
		 */
		// @formatter:off
		/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - A - - 
		 * 4 - - - R - A -
		 * 3 - - - - A - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		void entrarElReyPorElOeste() {
			arbitro.colocarPiezas(
					new TipoPieza[] { TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, 
							TipoPieza.REY },
					new int[][] { { 2, 4 }, { 3, 5 }, { 4, 4}, { 3, 3 } }, Color.BLANCO);
			// necesitamos mover para capturar y que pueda conocer el último movimiento
			arbitro.mover(fabricarJugada(tablero, 3, 3, 3, 4));
		}
	}

}
