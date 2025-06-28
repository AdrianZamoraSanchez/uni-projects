package tafl.control.ardri.basico;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Timeout.ThreadMode.SEPARATE_THREAD;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.ClassOrderer; // para ordenar la ejecución de las clases anidadas
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestClassOrder;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import tafl.control.ArbitroArdRi;
import tafl.excepcion.CoordenadasIncorrectasException;
import tafl.modelo.Celda;
import tafl.modelo.Pieza;
import tafl.modelo.Tablero;
import tafl.util.Color;
import tafl.util.Coordenada;
import tafl.util.TipoPieza;

/**
 * Comprobación del ArbitroArdRi de inicialización de la partida
 * colocando las piezas sobre el tablero.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@DisplayName("Tests del ArbitroArdRi sobre la inicialización de piezas.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
@TestClassOrder(ClassOrderer.OrderAnnotation.class) // ordenamos la ejecución por @Order
public class ArbitroInicializacionTest extends tafl.control.basico.ArbitroAbstractoInicializacionTest{

	/** Generación del árbitro para testing. */
	@BeforeEach
	void inicializar() {
		// Inyección de tablero para testing...
		tablero = new Tablero();
		arbitro = new ArbitroArdRi(tablero);
	}	

	/**
	 * Comprueba la correcta colocación de piezas con
	 * la configuración inicial por defecto del juego.
	 */
	@DisplayName("Tests sobre el estado inicial del árbitro colocando las piezas.")
	@Nested
	@Order(2)
	class ColocacionInicialDePiezas {

		/**
		 * Inicialización de piezas nuevas sobre el tablero.
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
		void colocarPiezas() {
			arbitro.colocarPiezasConfiguracionInicial();
		}

		/**
		 * Comprueba el número de piezas iniciales.
		 */
		@Test
		@DisplayName("Comprueba el número de piezas con la configuración inicial")
		void probarNumeroDePiezas() {
			assertAll("estado inicial tras colocar piezas ",
					() -> assertThat("El número de jugadas debería ser cero", arbitro.consultarNumeroJugada(), is(0)),
					() -> assertThat("Debería haber ocho piezas atacantes",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(16)),
					() -> assertThat("Debería haber cinco piezas defensoras",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(8)),
					() -> assertThat("Debería haber rey", tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)),
					() -> assertThat("Siempre empiezan atacantes.", arbitro.consultarTurno(), is(Color.NEGRO)));
		}

		/**
		 * Comprueba que se colocan piezas atacantes correctamente.
		 * 
		 * @param pieza      pieza
		 * @param coordenada coordenada
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@DisplayName("Comprobar piezas atacantes")
		@ParameterizedTest
		@MethodSource("tafl.control.TestUtil#piezaYCoordenadaDeAtacanteArdRiProvider")
		void comprobarAtacantes(Pieza pieza, Coordenada coordenada) throws CoordenadasIncorrectasException {
			Celda celda = tablero.consultarCelda(coordenada);
			assertAll("comprobar piezas atacantes", () -> assertThat("Celda vacia", celda.estaVacia(), is(false)),
					() -> assertThat("Color de pieza colocada incorrecta", celda.consultarColorDePieza(),
							is(Color.NEGRO)),
					() -> assertThat("Tipo de pieza incorrecto", celda.consultarPieza().consultarTipoPieza(),
							is(TipoPieza.ATACANTE)));
		}

		/**
		 * Comprueba que se colocan piezas defensoras correctamente.
		 * 
		 * @param pieza      pieza
		 * @param coordenada coordenada
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@DisplayName("Comprobar piezas defensoras")
		@ParameterizedTest
		@MethodSource("tafl.control.TestUtil#piezaYCoordenadaDeDefensorArdRiProvider")
		void comprobarDefensores(Pieza pieza, Coordenada coordenada) throws CoordenadasIncorrectasException {
			Celda celda = tablero.consultarCelda(coordenada);
			assertAll("comprobar piezas atacantes", () -> assertThat("Celda vacia", celda.estaVacia(), is(false)),
					() -> assertThat("Color de pieza colocada incorrecta", celda.consultarColorDePieza(),
							is(Color.BLANCO)),
					() -> assertThat("Tipo de pieza incorrecto", celda.consultarPieza().consultarTipoPieza(),
							is(TipoPieza.DEFENSOR)));
		}

		/**
		 * Comprueba que el rey está bien colocado.
		 * 
		 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		@DisplayName("Comprobar que el rey está bien colocado.")
		@Test
		void comprobarColocacionDelRey() throws CoordenadasIncorrectasException {
			Celda celda = tablero.consultarCelda(new Coordenada(3, 3));
			assertThat("El rey no está bien colocado.", celda.consultarPieza().consultarTipoPieza(), is(TipoPieza.REY));
		}
	}
}
