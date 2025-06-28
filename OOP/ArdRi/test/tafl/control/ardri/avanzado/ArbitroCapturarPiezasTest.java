package tafl.control.ardri.avanzado;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertAll;
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
import tafl.util.Coordenada;
import tafl.util.TipoPieza;


/**
 * Comprobación del ArbitroArdRi de captura de piezas en distintas condiciones.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@DisplayName("Tests del ArbitroArdRi sobre capturas de piezas.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public class ArbitroCapturarPiezasTest extends tafl.control.avanzado.ArbitroAbstractoCapturarPiezasTest {

	/** Generación del árbitro para testing. */
	@BeforeEach
	void inicializar() {
		// Inyección de tablero para testing...
		tablero = new Tablero();
		arbitro = new ArbitroArdRi(tablero);

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
		 * 7 - - A A A - -  
		 * 6 - - - A - - - 
		 * 5 A - D D D - A 
		 * 4 A A D R D A A 
		 * 3 A - D D D - A
		 * 2 - - - A - - - 
		 * 1 - - A A A - -
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
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarCapturaAtacanteContraProvinciaEnEsquinaInferiorDerechaEnHorizontal() throws CoordenadasIncorrectasException {
			arbitro.mover(fabricarJugada(tablero, 6, 4, 6, 5)); // atacante
			arbitro.mover(fabricarJugada(tablero, 3, 4, 6, 4)); // defensor (saltando dos aunque no sea legal)
			arbitro.realizarCapturasTrasMover();
			assertAll("captura de pieza atacante",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(6, 5)).estaVacia(), is(true)),
					() -> assertThat("No debería capturarse una pieza defensora.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(8)),
					() -> assertThat("Debería capturarse una pieza atacante.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(15)));
		}

		/**
		 * Comprueba la captura de atacante contra provincia en esquina superior
		 * derecha en horizontal.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarCapturaAtacanteContraProvinciaEnEsquinaSuperiorDerechaEnHorizontal() throws CoordenadasIncorrectasException {
			arbitro.mover(fabricarJugada(tablero, 0, 4, 0, 5)); // atacante
			arbitro.mover(fabricarJugada(tablero, 3, 4, 0, 4)); // defensor (saltando dos aunque no sea legal)

			arbitro.realizarCapturasTrasMover();
			assertAll("captura de pieza atacante",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(0, 5)).estaVacia(), is(true)),
					() -> assertThat("No debería capturarse una pieza defensora.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(8)),
					() -> assertThat("Debería capturarse una pieza atacante.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(15)));
		}

		/**
		 * Comprueba la captura de atacante contra provincia en esquina inferior 
		 * izquierda en horizontal.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarCapturaAtacanteContraProvinciaEnEsquinaInferiorIzquierdaEnHorizontal() throws CoordenadasIncorrectasException {
			arbitro.mover(fabricarJugada(tablero, 6, 2, 6, 1)); // atacante
			arbitro.mover(fabricarJugada(tablero, 4, 2, 6, 2)); // defensor (saltando dos aunque no sea legal)
			arbitro.realizarCapturasTrasMover();
			assertAll("captura de pieza atacante",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(6, 1)).estaVacia(), is(true)),
					() -> assertThat("No debería capturarse una pieza defensora.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(8)),
					() -> assertThat("Debería capturarse una pieza atacante.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(15)));
		}

		/**
		 * Comprueba la captura de atacante contra provincia en esquina superior
		 * izquierda en horizontal.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarCapturaAtacanteContraProvinciaEnEsquinaSuperiorIzquierdaEnHorizontal() throws CoordenadasIncorrectasException {
			arbitro.mover(fabricarJugada(tablero, 0, 2, 0, 1)); // atacante
			arbitro.mover(fabricarJugada(tablero, 2, 2, 0, 2)); // defensor (saltando dos aunque no sea legal)
			arbitro.realizarCapturasTrasMover();
			assertAll("captura de pieza atacante",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(0, 1)).estaVacia(), is(true)),
					() -> assertThat("No debería capturarse una pieza defensora.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(8)),
					() -> assertThat("Debería capturarse una pieza atacante.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(15)));
		}

		/**
		 * Comprueba la captura de atacante contra provincia en esquina superior 
		 * izquierda en vertical.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarCapturaAtacanteContraProvinciaEnEsquinaSuperiorIzquierdaEnVertical() throws CoordenadasIncorrectasException {
			arbitro.mover(fabricarJugada(tablero, 2, 0, 1, 0)); // atacante
			arbitro.mover(fabricarJugada(tablero, 2, 2, 2, 0)); // defensor (saltando dos aunque no sea legal)
			arbitro.realizarCapturasTrasMover();
			assertAll("captura de pieza atacante",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(1, 0)).estaVacia(), is(true)),
					() -> assertThat("No debería capturarse una pieza defensora.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(8)),
					() -> assertThat("Debería capturarse una pieza atacante.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(15)));
		}

		/**
		 * Comprueba la captura de atacante contra provincia en esquina superior
		 * derecha en vertical.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test		
		void comprobarCapturaAtacanteContraProvinciaEnEsquinaSuperiorDerechaEnVertical() throws CoordenadasIncorrectasException {
			arbitro.mover(fabricarJugada(tablero, 2, 6, 1, 6)); // atacante
			arbitro.mover(fabricarJugada(tablero, 2, 4, 2, 6)); // defensor (saltando dos aunque no sea legal)
			arbitro.realizarCapturasTrasMover();
			assertAll("captura de pieza atacante",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(1, 6)).estaVacia(), is(true)),
					() -> assertThat("No debería capturarse una pieza defensora.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(8)),
					() -> assertThat("Debería capturarse una pieza atacante.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(15)));
		}

		/**
		 * Comprueba la captura de atacante contra provincia en esquina inferior 
		 * izquierda en vertical.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarCapturaAtacanteContraProvinciaEnEsquinaInferiorIzquierdaEnVertical() throws CoordenadasIncorrectasException {
			arbitro.mover(fabricarJugada(tablero, 4, 0, 5, 0)); // atacante
			arbitro.mover(fabricarJugada(tablero, 4, 2, 4, 0)); // defensor (saltando dos aunque no sea legal)
			arbitro.realizarCapturasTrasMover();
			assertAll("captura de pieza atacante",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(5, 0)).estaVacia(), is(true)),
					() -> assertThat("No debería capturarse una pieza defensora.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(8)),
					() -> assertThat("Debería capturarse una pieza atacante.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(15)));
		}

		/**
		 * Comprueba la captura de atacante contra provincia en esquina inferior 
		 * derecha en vertical.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarCapturaAtacanteContraProvinciaEnEsquinaInferiorDerechaEnVertical() throws CoordenadasIncorrectasException {
			arbitro.mover(fabricarJugada(tablero, 4, 6, 5, 6)); // atacante
			arbitro.mover(fabricarJugada(tablero, 4, 4, 4, 6)); // defensor (saltando dos aunque no sea legal)
			arbitro.realizarCapturasTrasMover();
			assertAll("captura de pieza atacante",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(5, 6)).estaVacia(), is(true)),
					() -> assertThat("No debería capturarse una pieza defensora.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(8)),
					() -> assertThat("Debería capturarse una pieza atacante.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(15)));
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
		 * 7 - - A A A - -  
		 * 6 - - - A - - - 
		 * 5 A - D D D - A 
		 * 4 A A D R D A A 
		 * 3 A - D D D - A
		 * 2 - - - A - - - 
		 * 1 - - A A A - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		@BeforeEach
		void inicializar() {
			arbitro.colocarPiezasConfiguracionInicial();
		}

		/**
		 * Comprueba la captura de defensor en cuadrante izquierdo superior en vertical.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarCapturaDefensorEnCuadranteIzquierdoSuperiorEnVertical() throws CoordenadasIncorrectasException {
			arbitro.mover(fabricarJugada(tablero, 2, 2, 2, 1)); // defensor
			arbitro.mover(fabricarJugada(tablero, 1, 3, 1, 1)); // atacante (saltando dos aunque no sea legal)
			arbitro.realizarCapturasTrasMover();

			assertAll("captura de pieza del defensor",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(2, 1)).estaVacia(), is(true)),
					() -> assertThat("Deberían estar el número inicial de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(16)),
					() -> assertThat("Debería capturarse una pieza del defensor.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(7)));
		}

		/**
		 * Comprueba la captura de defensor en cuadrante derecho superior en vertical.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarCapturaDefensorEnCuadranteDerechoSuperiorEnVertical() throws CoordenadasIncorrectasException {
			arbitro.mover(fabricarJugada(tablero, 2, 4, 2, 5)); // defensor
			arbitro.mover(fabricarJugada(tablero, 1, 3, 1, 5)); // atacante (saltando dos aunque no sea legal)
			arbitro.realizarCapturasTrasMover();

			assertAll("captura de pieza del defensor",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(2, 5)).estaVacia(), is(true)),
					() -> assertThat("Deberían estar el número inicial de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(16)),
					() -> assertThat("Debería capturarse una pieza del defensor.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(7)));
		}

		/**
		 * Comprueba la captura de defensor en cuadrante izquierdo inferior en vertical.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarCapturaDefensorEnCuadranteIzquierdoInferiorEnVertical() throws CoordenadasIncorrectasException {
			arbitro.mover(fabricarJugada(tablero, 4, 2, 4, 1)); // defensor
			arbitro.mover(fabricarJugada(tablero, 5, 3, 5, 1)); // atacante (saltando dos aunque no sea legal)
			arbitro.realizarCapturasTrasMover();

			assertAll("captura de pieza del defensor",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(4, 1)).estaVacia(), is(true)),
					() -> assertThat("Deberían estar el número inicial de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(16)),
					() -> assertThat("Debería capturarse una pieza del defensor.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(7)));
		}

		/**
		 * Comprueba la captura de defensor en cuadrante derecho inferior en vertical.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarCapturaDefensorEnCuadranteDerechoInferiorEnVertical() throws CoordenadasIncorrectasException {
			arbitro.mover(fabricarJugada(tablero, 4, 4, 4, 5)); // defensor
			arbitro.mover(fabricarJugada(tablero, 5, 3, 5, 5)); // atacante (saltando dos aunque no sea legal)
			arbitro.realizarCapturasTrasMover();

			assertAll("captura de pieza del defensor",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(4, 5)).estaVacia(), is(true)),
					() -> assertThat("Deberían estar el número inicial de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(16)),
					() -> assertThat("Debería capturarse una pieza del defensor.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(7)));
		}

		/**
		 * Comprueba la captura de defensor en cuadrante izquierdo superior en horizontal.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarCapturaDefensorEnCuadranteIzquierdoSuperiorEnHorizontal() throws CoordenadasIncorrectasException {
			arbitro.mover(fabricarJugada(tablero, 2, 2, 1, 2)); // defensor
			arbitro.mover(fabricarJugada(tablero, 3, 1, 1, 1)); // atacante (saltando dos aunque no sea legal)
			arbitro.realizarCapturasTrasMover(); 

			assertAll("captura de pieza del defensor",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(1, 2)).estaVacia(), is(true)),
					() -> assertThat("Deberían estar el número inicial de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(16)),
					() -> assertThat("Debería capturarse una pieza del defensor.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(7)));
		}

		/**
		 * Comprueba la captura de defensor en cuadrante derecho superior en horizontal.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarCapturaDefensorEnCuadranteDerechoSuperiorEnHorizontal() throws CoordenadasIncorrectasException {
			arbitro.mover(fabricarJugada(tablero, 2, 4, 1, 4)); // defensor
			arbitro.mover(fabricarJugada(tablero, 3, 5, 1, 5)); // atacante (saltando dos aunque no sea legal)
			arbitro.realizarCapturasTrasMover();

			assertAll("captura de pieza del defensor",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(1, 4)).estaVacia(), is(true)),
					() -> assertThat("Deberían estar el número inicial de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(16)),
					() -> assertThat("Debería capturarse una pieza del defensor.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(7)));
		}

		/**
		 * Comprueba la captura de defensor en cuadrante izquierdo inferior en horizontal.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarCapturaDefensorEnCuadranteIzquierdoInferiorEnHorizontal() throws CoordenadasIncorrectasException {
			arbitro.mover(fabricarJugada(tablero, 4, 2, 5, 2)); // defensor
			arbitro.mover(fabricarJugada(tablero, 3, 1, 5, 1)); // atacante (saltando dos aunque no sea legal)
			arbitro.realizarCapturasTrasMover();

			assertAll("captura de pieza del defensor",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(5, 2)).estaVacia(), is(true)),
					() -> assertThat("Deberían estar el número inicial de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(16)),
					() -> assertThat("Debería capturarse una pieza del defensor.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(7)));
		}

		/**
		 * Comprueba la captura de defensor en cuadrante derecho inferior en horizontal.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@Test
		void comprobarCapturaDefensorEnCuadranteDerechoInferiorEnHorizontal() throws CoordenadasIncorrectasException {
			arbitro.mover(fabricarJugada(tablero, 4, 4, 5, 4)); // defensor
			arbitro.mover(fabricarJugada(tablero, 3, 5, 5, 5)); // atacante (saltando dos aunque no sea legal)
			arbitro.realizarCapturasTrasMover();

			assertAll("captura de pieza del defensor",
					() -> assertThat("Debería estar vacía la celda con pieza capturada.",
							tablero.consultarCelda(new Coordenada(5, 4)).estaVacia(), is(true)),
					() -> assertThat("Deberían estar el número inicial de piezas atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(16)),
					() -> assertThat("Debería capturarse una pieza del defensor.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(7)));
		}
	}
}
