package brandubh.control.basico;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.CoreMatchers.nullValue;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertAll;

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

import brandubh.control.Arbitro;
import brandubh.modelo.Celda;
import brandubh.modelo.Pieza;
import brandubh.modelo.Tablero;
import brandubh.util.Color;
import brandubh.util.Coordenada;
import brandubh.util.TipoPieza;

/**
 * Comprobación de inicialización de la partida
 * colocando las piezas sobre el tablero.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@DisplayName("Tests del Arbitro sobre la inicialización de piezas.")
@Timeout(value = 2, unit = TimeUnit.SECONDS) // Time out global para todos los tests salvo los de ciclo de vida
@TestClassOrder(ClassOrderer.OrderAnnotation.class) // ordenamos la ejecución por @Order
public class ArbitroInicializacionTest {

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
	 * Comprobación del estado inicial del árbitro sin piezas colocadas.
	 * 
	 * Este conjunto de pruebas se centra en verificar el estado del árbitro justo
	 * después de instanciar pero antes de colocar las piezas.
	 * 
	 * @author <a href="mailto:rmartico@ubu.es">Raúl Marticorena Sánchez</a>
	 * @version 1.0 20230708
	 */
	@DisplayName("Tests sobre el estado inicial del árbitro antes de colocar piezas")
	@Nested
	@Order(1)
	class InicioDePartida {

		/**
		 * Comprobacion de inicialización correcta del tablero, sin colocar ninguna
		 * pieza, con un tablero vacío y sin turno incialmente.
		 */
		// @formatter:off
		 /* Partimos de un tablero vacío como el que se muestra:	
		 * 7 - - - - - - -  
		 * 6 - - - - - - - 
		 * 5 - - - - - - - 
		 * 4 - - - - - - - 
		 * 3 - - - - - - -
		 * 2 - - - - - - - 
		 * 1 - - - - - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		@DisplayName("Comprueba el estado inicial del tablero vacío")
		@Test
		void comprobarEstadoInicialAntesDeRellenarTablero() {
			assertAll("estado inicial ",
					() -> assertThat("El número de jugadas debería ser cero", arbitro.consultarNumeroJugada(), is(0)),
					() -> assertThat("No debería haber piezas atacantes",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(0)),
					() -> assertThat("No debería haber piezas defensoras",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(0)),
					() -> assertThat("No debería haber rey", tablero.consultarNumeroPiezas(TipoPieza.REY), is(0)),
					() -> assertThat("El turno se ha inicializado cuando debería valer nulo.", arbitro.consultarTurno(), is(nullValue())));
		}
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
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
					() -> assertThat("Debería haber cinco piezas defensoras",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
					() -> assertThat("Debería haber rey", tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)),
					() -> assertThat("Siempre empiezan atacantes.", arbitro.consultarTurno(), is(Color.NEGRO)));
		}

		/**
		 * Comprueba que se colocan piezas atacantes correctamente.
		 * 
		 * @param pieza      pieza
		 * @param coordenada coordenada
		 */
		@DisplayName("Comprueba piezas atacantes")
		@ParameterizedTest
		@MethodSource("brandubh.control.TestUtil#piezaYCoordenadaDeAtacanteProvider")
		void comprobarAtacantes(Pieza pieza, Coordenada coordenada) {
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
		 */
		@DisplayName("Comprueba piezas defensoras")
		@ParameterizedTest
		@MethodSource("brandubh.control.TestUtil#piezaYCoordenadaDeDefensorProvider")
		void comprobarDefensores(Pieza pieza, Coordenada coordenada) {
			Celda celda = tablero.consultarCelda(coordenada);
			assertAll("comprobar piezas atacantes", () -> assertThat("Celda vacia", celda.estaVacia(), is(false)),
					() -> assertThat("Color de pieza colocada incorrecta", celda.consultarColorDePieza(),
							is(Color.BLANCO)),
					() -> assertThat("Tipo de pieza incorrecto", celda.consultarPieza().consultarTipoPieza(),
							is(TipoPieza.DEFENSOR)));
		}

		/**
		 * Comprueba que el rey está bien colocado.
		 */
		@DisplayName("Comprueba que el rey está bien colocado.")
		@Test
		void comprobarColocacionDelRey() {
			Celda celda = tablero.consultarCelda(new Coordenada(3, 3));
			assertThat("El rey no está bien colocado.", celda.consultarPieza().consultarTipoPieza(), is(TipoPieza.REY));
		}
	}

	/**
	 * Comprueba la colocación de piezas ad-hoc, sin tener que seguir 
	 * la configuración por defecto nicial del juego.
	 * 
	 */
	@DisplayName("Tests sobre el estado inicial del árbitro colocando las piezas ad-hoc.")
	@Nested
	@Order(3)
	class ColocacionAdHocDePiezas {

		/**
		 * Comprueba la colocación de todas las piezas en posición diferente.
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - A - - -  
		 * 6 - - A - - D - 
		 * 5 - A - R D - - 
		 * 4 A - D -  - A 
		 * 3 - D - - - A -
		 * 2 - - - - A - - 
		 * 1 - - - A - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		@Test
		@DisplayName("Comprueba la colocación de todas las piezas en posición diferente.")
		void probarColocandoTodasLasPiezasAdHoc() {
			arbitro.colocarPiezas(new TipoPieza[] {TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE,
					TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE,
					TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR, TipoPieza.DEFENSOR,
					TipoPieza.REY}, 
					
					new int[][] {{0, 3}, {1, 2}, {2, 1}, {3, 0}, {3, 6}, {4, 5}, {5, 4}, {6, 3},
						// defensores
						{1, 5}, {2, 4}, {3, 2}, {4, 1},
						// rey
						{2, 3}
					},
					Color.NEGRO);
			
			assertAll("estado inicial tras colocar piezas sin seguir el orden habitual ",
					() -> assertThat("El número de jugadas debería ser cero", arbitro.consultarNumeroJugada(), is(0)),
					() -> assertThat("Debería haber ocho piezas atacantes",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(8)),
					() -> assertThat("Debería haber cinco piezas defensoras",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
					() -> assertThat("Debería haber rey", tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)),
					() -> assertThat("La pieza atacante está bien colocada.", tablero.consultarCelda(new Coordenada(0,3)).consultarPieza().consultarTipoPieza(), is(TipoPieza.ATACANTE)),
					() -> assertThat("La pieza atacante está bien colocada.", tablero.consultarCelda(new Coordenada(1,2)).consultarPieza().consultarTipoPieza(), is(TipoPieza.ATACANTE)),
					() -> assertThat("La pieza atacante está bien colocada.", tablero.consultarCelda(new Coordenada(2,1)).consultarPieza().consultarTipoPieza(), is(TipoPieza.ATACANTE)),
					() -> assertThat("La pieza atacante está bien colocada.", tablero.consultarCelda(new Coordenada(3,0)).consultarPieza().consultarTipoPieza(), is(TipoPieza.ATACANTE)),
					() -> assertThat("La pieza atacante está bien colocada.", tablero.consultarCelda(new Coordenada(3,6)).consultarPieza().consultarTipoPieza(), is(TipoPieza.ATACANTE)),
					() -> assertThat("La pieza atacante está bien colocada.", tablero.consultarCelda(new Coordenada(4,5)).consultarPieza().consultarTipoPieza(), is(TipoPieza.ATACANTE)),
					() -> assertThat("La pieza atacante está bien colocada.", tablero.consultarCelda(new Coordenada(5,4)).consultarPieza().consultarTipoPieza(), is(TipoPieza.ATACANTE)),
					() -> assertThat("La pieza atacante está bien colocada.", tablero.consultarCelda(new Coordenada(6,3)).consultarPieza().consultarTipoPieza(), is(TipoPieza.ATACANTE)),

					() -> assertThat("La pieza defensora está bien colocada.", tablero.consultarCelda(new Coordenada(1,5)).consultarPieza().consultarTipoPieza(), is(TipoPieza.DEFENSOR)),
					() -> assertThat("La pieza defensora está bien colocada.", tablero.consultarCelda(new Coordenada(2,4)).consultarPieza().consultarTipoPieza(), is(TipoPieza.DEFENSOR)),
					() -> assertThat("La pieza defensora está bien colocada.", tablero.consultarCelda(new Coordenada(3,2)).consultarPieza().consultarTipoPieza(), is(TipoPieza.DEFENSOR)),
					() -> assertThat("La pieza defensora está bien colocada.", tablero.consultarCelda(new Coordenada(4,1)).consultarPieza().consultarTipoPieza(), is(TipoPieza.DEFENSOR)),
					
					() -> assertThat("El rey está bien colocado.", tablero.consultarCelda(new Coordenada(2,3)).consultarPieza().consultarTipoPieza(), is(TipoPieza.REY)),

					() -> assertThat("Siempre empiezan atacantes.", arbitro.consultarTurno(), is(Color.NEGRO)));
			
		}
		
		/**
		 * Comprueba la colocación de la mitad de piezas en posición diferente y el rey.
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - A - - -  
		 * 6 - - - - - D - 
		 * 5 - A - - - - - 
		 * 4 - - - - - - - 
		 * 3 - D - - - A -
		 * 2 - - - - - - - 
		 * 1 - - - A - R -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		@Test
		@DisplayName("Comprueba la colocación de la mitad de piezas en posición diferente.")
		void probarColocandoLaMitadDeLasPiezasAdHoc() {
			arbitro.colocarPiezas(new TipoPieza[] {
					TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE,
					TipoPieza.DEFENSOR, TipoPieza.DEFENSOR,
					TipoPieza.REY}, 
					
					new int[][] {{0, 3}, {2, 1}, {4, 5}, {6, 3},
						// defensores
						{1, 5}, {4, 1},
						// rey
						{6, 5}
					},
					Color.NEGRO);
			
			assertAll("estado inicial tras colocar piezas sin seguir el orden habitual ",
					() -> assertThat("El número de jugadas debería ser cero", arbitro.consultarNumeroJugada(), is(0)),
					() -> assertThat("Debería haber ocho piezas atacantes",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(4)),
					() -> assertThat("Debería haber cinco piezas defensoras",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(2)),
					() -> assertThat("Debería haber rey", tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)),
					() -> assertThat("La pieza atacante está bien colocada.", tablero.consultarCelda(new Coordenada(0,3)).consultarPieza().consultarTipoPieza(), is(TipoPieza.ATACANTE)),
					() -> assertThat("La pieza atacante está bien colocada.", tablero.consultarCelda(new Coordenada(2,1)).consultarPieza().consultarTipoPieza(), is(TipoPieza.ATACANTE)),
					() -> assertThat("La pieza atacante está bien colocada.", tablero.consultarCelda(new Coordenada(4,5)).consultarPieza().consultarTipoPieza(), is(TipoPieza.ATACANTE)),
					() -> assertThat("La pieza atacante está bien colocada.", tablero.consultarCelda(new Coordenada(6,3)).consultarPieza().consultarTipoPieza(), is(TipoPieza.ATACANTE)),

					() -> assertThat("La pieza defensora está bien colocada.", tablero.consultarCelda(new Coordenada(1,5)).consultarPieza().consultarTipoPieza(), is(TipoPieza.DEFENSOR)),
					() -> assertThat("La pieza defensora está bien colocada.", tablero.consultarCelda(new Coordenada(4,1)).consultarPieza().consultarTipoPieza(), is(TipoPieza.DEFENSOR)),
					
					() -> assertThat("El rey está bien colocado.", tablero.consultarCelda(new Coordenada(6,5)).consultarPieza().consultarTipoPieza(), is(TipoPieza.REY)),

					() -> assertThat("Siempre empiezan atacantes.", arbitro.consultarTurno(), is(Color.NEGRO)));
			
		}
	}
}
