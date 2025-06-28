package tafl.control.basico;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.CoreMatchers.nullValue;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Timeout.ThreadMode.SEPARATE_THREAD;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.ClassOrderer; // para ordenar la ejecución de las clases anidadas
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestClassOrder;
import org.junit.jupiter.api.Timeout;

import tafl.control.Arbitro;
import tafl.excepcion.CoordenadasIncorrectasException;
import tafl.modelo.Tablero;
import tafl.util.Color;
import tafl.util.Coordenada;
import tafl.util.TipoPieza;

/**
 * Comprobación de inicialización de la partida
 * colocando las piezas sobre el tablero.
 * Aplicable a ambos tipos de juego.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@DisplayName("Tests del arbitro sobre la inicialización de piezas.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
@TestClassOrder(ClassOrderer.OrderAnnotation.class) // ordenamos la ejecución por @Order
public abstract class ArbitroAbstractoInicializacionTest {

	/** Árbitro de testing. */
	protected Arbitro arbitro;

	/** Tablero de testing. */
	protected Tablero tablero;	

	/**
	 * Comprobación del estado inicial del árbitro sin piezas colocadas. 
	 * Este conjunto de pruebas se centra en verificar el estado del árbitro justo
	 * después de instanciar pero antes de colocar las piezas.
	 * 
	 * @author <a href="mailto:rmartico@ubu.es">Raúl Marticorena Sánchez</a>
	 * @version 1.0 20230708
	 * @see tafl.control.Arbitro#colocarPiezas(TipoPieza[], int[][], Color)
	 */
	@DisplayName("Tests sobre el estado inicial del árbitro antes de colocar piezas")
	@Nested
	@Order(1)
	class InicioDePartida {

		/**
		 * Comprobacion de inicialización correcta del tablero, sin colocar ninguna
		 * pieza, con un tablero vacío y sin turno incialmente. Aplicable a ambos tipos de juego.
		 * 
		 * @version 1.0.0.1
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
					() -> assertThat("El turno se ha inicializado cuando debería valer nulo.", arbitro.consultarTurno(), is(nullValue()))); // Changed
		}
	}


	/**
	 * Comprueba la colocación de piezas ad-hoc, sin tener que seguir 
	 * la configuración por defecto nicial del juego. Aplicable a ambos tipos de juego.
	 * 
	 */
	@DisplayName("Tests sobre el estado inicial del árbitro colocando las piezas ad-hoc.")
	@Nested
	@Order(3)
	class ColocacionAdHocDePiezas {

		/**
		 * Comprueba la colocación de todas las piezas en posición diferente.
		 * 
	     * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
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
		@DisplayName("Comprueba la colocación de 5 defensores y 8 atacantes en posiciones aleatorias.")
		void probarColocandoTodasLasPiezasAdHoc() throws CoordenadasIncorrectasException {
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
		 * Comprueba la colocación de 3 defensores y 4 atacantes en posiciones aleatorias.
		 * 
	     * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
		 */
		// @formatter:off
		/* Rellenaremos el tablero tal y como se muestra:	
		 * 7 - - - A - - -  
		 * 6 - - - - - D - 
		 * 5 - A - - - - - 
		 * 4 - - - - - - - 
		 * 3 - D - - - A -
		 * 2 - - - - - R - 
		 * 1 - - - A - - -
		 *   a b c d e f g   
		 */
		 // @formatter:on
		@Test
		@DisplayName("Comprueba la colocación de 3 defensores y 4 atacantes en posiciones aleatorias.")
		void probarColocandoLaMitadDeLasPiezasAdHoc()  throws CoordenadasIncorrectasException {
			arbitro.colocarPiezas(new TipoPieza[] {
					TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE, TipoPieza.ATACANTE,
					TipoPieza.DEFENSOR, TipoPieza.DEFENSOR,
					TipoPieza.REY}, 
					
					new int[][] {{0, 3}, {2, 1}, {4, 5}, {6, 3},
						// defensores
						{1, 5}, {4, 1},
						// rey
						{5, 5}
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
					
					() -> assertThat("El rey está bien colocado.", tablero.consultarCelda(new Coordenada(5,5)).consultarPieza().consultarTipoPieza(), is(TipoPieza.REY)),

					() -> assertThat("Siempre empiezan atacantes.", arbitro.consultarTurno(), is(Color.NEGRO)));
			
		}
	}
}
