package brandubh.modelo;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.ClassOrderer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestClassOrder;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import brandubh.util.Coordenada;
import brandubh.util.TipoCelda;
import brandubh.util.TipoPieza;

/**
 * Tests sobre el Tablero.
 * 
 * Se ordenan los tests agrupados por subclases (ver uso de anotación @Order). 
 * El orden de ejecución sugiere el orden en el que se deberían ir implementando 
 * los métodos.
 * 
 * @author <a href="mailto:rmartico@ubu.es">Raúl Marticorena Sánchez</a>
 * @version 1.0 20230707
 * 
 */
@DisplayName("Tests sobre Tablero (depende de implementaciones reales de Pieza, Celda)")
@Timeout(value = 2, unit = TimeUnit.SECONDS) // Time out global para todos los tests salvo los de ciclo de vida
@TestClassOrder(ClassOrderer.OrderAnnotation.class) // ordenamos la ejecución por @Order
public class TableroTest {

	/** Número de celdas en el Brandubh. */
	private static final int TOTAL_CELDAS_BRANDUBH = 49;

	/** Tablero de testing. */
	private Tablero tablero;

	/** Inicializa valores para cada test. */
	@BeforeEach
	@Timeout(value = 1000, unit = TimeUnit.MILLISECONDS)
	void inicializar() {
		tablero = new Tablero();
	}

	/**
	 * Coloca nueve piezas en aspa sobre el tablero.
	 * 
	 * Método de utilida utilizado en varios tests.
	 * 
	 * <pre>
	 *  
	 * 	7 A - - - - - A
	 *	6 - D - - - D -
	 *	5 - - - - - - -
	 *	4 - - - R - - -
	 *	3 - - - - - - -
	 *  2 - D - - - D -
	 *	1 A - - - - - A
	 *    a b c d e f g
	 * </pre>
	 * 
	 */
	private void colocarNuevePiezasEnAspa() {
		tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(0, 0));
		tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(0, 6));
		tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(6, 0));
		tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(6, 6));
		tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(1, 1));
		tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(1, 5));
		tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(5, 1));
		tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(5, 5));
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(3, 3));
	}

	/**
	 * Estado inicial básico.
	 * 
	 * @see brandubh.modelo.Tablero#consultarNumeroFilas()
	 * @see brandubh.modelo.Tablero#consultarNumeroColumnas()
	 */
	@Nested
	@DisplayName("Tests de estado inicial básico.")
	@Order(1)
	class EstadoInicialBasico {

		/**
		 * Comprueba que el tablero se inicializa con el número de filas y columnas
		 * correcto.
		 */
		@Test
		@DisplayName("Comprueba estado inicial básico.")
		void comprobarEstadoInicial() {
			assertAll("estado inicial básico",
					() -> assertThat("Número de filas incorrecto.", tablero.consultarNumeroFilas(), is(7)),
					() -> assertThat("Número de columnas incorrecto.", tablero.consultarNumeroFilas(), is(7)));
		}
	}

	/**
	 * Consulta de celdas.
	 * 
	 * @see brandubh.modelo.Tablero#obtenerCelda(Coordenada)
	 * @see brandubh.modelo.Tablero#consultarCelda(Coordenada)
	 */
	@Nested
	@DisplayName("Tests de consulta de celdas.")
	@Order(2)
	class ConsultaCeldas {

		/**
		 * Comprueba la obtención de celda en posiciones correctas del tablero vacío.
		 * 
		 * @param fila fila
		 * @param columna columna
		 */
		@DisplayName("Comprueba la obtención de celda en coordenadass correctas del tablero")
		@ParameterizedTest
		@MethodSource("brandubh.modelo.TestUtil#proveerCoordenadas")
		void comprobarObtenciónDeCeldaEnPosicionesCorrectas(int fila, int columna) {
			// given
			Coordenada coordenada = new Coordenada(fila, columna);
			// when
			// then
			assertAll("valores en la celda obtenida",
					() -> assertNotNull("La celda sí debería estar contenida en el tablero, no debe devolver un nulo",
							tablero.obtenerCelda(coordenada)),
					() -> assertThat("Las coordenadas no son correctas.",
							tablero.obtenerCelda(coordenada).consultarCoordenada(), is(coordenada)),
					() -> assertThat("La celda debería estar vacía.", tablero.obtenerCelda(coordenada).estaVacia(),
							is(true)));
		}

		/**
		 * Comprueba la consulta de celda en posiciones correctas del tablero vacío.
		 * 
		 * @param fila fila
		 * @param columna columna
		 */
		@DisplayName("Comprueba la consulta de celda en coordenadass correctas del tablero")
		@ParameterizedTest
		@MethodSource("brandubh.modelo.TestUtil#proveerCoordenadas")
		void comprobarConsultaDeCeldaEnPosicionesCorrectas(int fila, int columna) {
			// given
			Coordenada coordenada = new Coordenada(fila, columna);
			// when
			// then
			assertAll("valores en la celda obtenida",
					() -> assertNotNull("La celda sí debería estar contenida en el tablero, no debe devolver un nulo",
							tablero.consultarCelda(coordenada)),
					() -> assertThat("Las coordenadas no son correctas.",
							tablero.consultarCelda(coordenada).consultarCoordenada(), is(coordenada)),
					() -> assertThat("La celda debería estar vacía.", tablero.consultarCelda(coordenada).estaVacia(),
							is(true)));
		}

		/**
		 * Comprueba que la obtención y consulta de celda en posiciones correctas del
		 * tablero vacío devuelve celdas iguales en contenido pero con diferente
		 * identidad.
		 * 
		 * @param fila fila
		 * @param columna columna
		 */
		@DisplayName("Comprueba que la obtención y consulta de celda en coordenadas correctas devuelve celdas iguales con diferente identidad")
		@ParameterizedTest
		@MethodSource("brandubh.modelo.TestUtil#proveerCoordenadas")
		void comprobarObtencionYConsultaDeCeldaEnPosicionesCorrectasDevuelveCeldasIgualesEnContenidoYDiferenteIdentidad(
				int fila, int columna) {
			// given
			Coordenada coordenada = new Coordenada(fila, columna);
			// when
			// then
			assertAll("valores en la celda obtenida y consultada",
					() -> assertNotSame(tablero.obtenerCelda(coordenada), tablero.consultarCelda(coordenada),
							"La celda obtenida y consultada son el mismo objeto."),
					() -> assertThat("Las celdas deberían ser iguales en contenido.", tablero.obtenerCelda(coordenada),
							is(tablero.consultarCelda(coordenada))));
		}

		/**
		 * Comprueba que devuelve todas las celdas con independencia del orden.
		 */
		@DisplayName("Comprueba que la consulta de todas las celdas devuelve efectivamente todas (con independencia del orden)")
		@Test
		void comprobarConsultarCeldas() {
			Celda[] todas = tablero.consultarCeldas();
			int encontrada = 0;
			for (int i = 0; i < tablero.consultarNumeroFilas(); i++) {
				for (int j = 0; j < tablero.consultarNumeroColumnas(); j++) {
					Celda celda = tablero.consultarCelda(new Coordenada(i, j));
					for (Celda celdaAux : todas) {
						if (celdaAux.equals(celda)) {
							encontrada++;
							break;
						}
					}
				}
			}
			assertThat("No devuelve todas las celdas.", encontrada, is(TOTAL_CELDAS_BRANDUBH));
		}

		/**
		 * Comprueba que los tipos de celda en las esquinas son de tipo provincia.
		 */
		@Test
		@DisplayName("Comprueba que los tipos de celda en las esquinas son provincias.")
		void comprobarTiposDeCeldasEnLasEsquinasSonProvincias() {
			assertAll("comprobación de que las esquinas son provincias",
					() -> assertThat("La celda debería ser una provincia.",
							tablero.obtenerCelda(new Coordenada(0, 0)).consultarTipoCelda(), is(TipoCelda.PROVINCIA)),
					() -> assertThat("La celda debería ser una provincia.",
							tablero.obtenerCelda(new Coordenada(0, 6)).consultarTipoCelda(), is(TipoCelda.PROVINCIA)),
					() -> assertThat("La celda debería ser una provincia.",
							tablero.obtenerCelda(new Coordenada(6, 0)).consultarTipoCelda(), is(TipoCelda.PROVINCIA)),
					() -> assertThat("La celda debería ser una provincia.",
							tablero.obtenerCelda(new Coordenada(6, 6)).consultarTipoCelda(), is(TipoCelda.PROVINCIA)));
		}

		/**
		 * Comprueba que el tipo de celda en el centro es un trono.
		 */
		@Test
		@DisplayName("Comprueba que el tipo de celda en el centro es un trono.")
		void comprobarTipoDeCeldaEnElCentroEsTrono() {
			assertAll("comprobación de que la celda en el centro es un trono",
					() -> assertThat("La celda debería ser una provincia.",
							tablero.obtenerCelda(new Coordenada(3, 3)).consultarTipoCelda(), is(TipoCelda.TRONO)));
		}
		
		/**
		 * Comprueba que el tipo de celdas normales son correctas.
		 */
		@Test
		@DisplayName("Comprueba el tipo de celdas normales.")
		void comprobar() {
			// inicializamos las posiciones de provincia y trono para buscar si la lista contiene
			// o no la coordenada
			List<Coordenada> coordenadas = Arrays.asList(new Coordenada(0,0), new Coordenada(0,6),
					new Coordenada(6,0), new Coordenada(6,6), new Coordenada(3,3));
			int numeroCeldasNormales = 0;
			for (int i = 0; i < tablero.consultarNumeroFilas(); i++) {
				for (int j = 0; j < tablero.consultarNumeroColumnas(); j++) {
					Celda celda = tablero.obtenerCelda(new Coordenada(i, j));
					if (!coordenadas.contains(celda.consultarCoordenada()) && // si no es una coordenada de provincia o trono
							celda.consultarTipoCelda() == TipoCelda.NORMAL) {
						numeroCeldasNormales++; // entonces sí es de tipo normal
					}
				}
			}
			assertThat("El número de celdas de tipo normal no es correcto.", numeroCeldasNormales, is(44));			
		}
	}

	/**
	 * Consulta de celdas contiguas a partir de una coordenada origen.
	 * 
	 * @see brandubh.modelo.Tablero#consultarCeldasContiguas(Coordenada)
	 * @see brandubh.modelo.Tablero#consultarCeldasContiguasEnHorizontal(Coordenada)
	 * @see brandubh.modelo.Tablero#consultarCeldasContiguasEnVertical(Coordenada)
	 */
	@Nested
	@DisplayName("Tests de consulta de celdas contiguas.")
	@Order(3)
	class ConsultaCeldasContiguas {

		/**
		 * Comprueba celdas contiguas en el centro del tablero.
		 */
		@DisplayName("Comprueba que se encuentran las cuatro celdas contiguas a celda en el centro del tablero.")
		@Test
		void comprobarCeldasContiguasEnCentroDelTablero() {
			// given
			Coordenada coordenada = new Coordenada(3, 3);
			// when
			Celda[] contiguas = tablero.consultarCeldasContiguas(coordenada); // celda consultada
			Celda[] esperadas = { // celdas contiguas
					tablero.consultarCelda(new Coordenada(2, 3)), tablero.consultarCelda(new Coordenada(4, 3)),
					tablero.consultarCelda(new Coordenada(3, 2)), tablero.consultarCelda(new Coordenada(3, 4)) };
			// then
			assertAll("", () -> assertThat("Debería tener cuatro celdas contiguas.", contiguas.length, is(4)),
					() -> assertThat("Debería contener las cuatro celdas esperadas en cualquier orden.",
							Arrays.asList(contiguas), containsInAnyOrder(esperadas)));
		}

		/**
		 * Comprueba celdas contiguas en la esquina superior izquierda del tablero.
		 */
		@DisplayName("Comprueba que se encuentran las dos celdas contiguas a celda en esquina superior izquierda del tablero.")
		@Test
		void comprobarCeldasContiguasEnEsquinaSuperiorIzquierdaDelTablero() {
			// given
			Coordenada coordenada = new Coordenada(0, 0);
			// when
			Celda[] contiguas = tablero.consultarCeldasContiguas(coordenada); // celda consultada
			Celda[] esperadas = { // celdas contiguas
					tablero.consultarCelda(new Coordenada(0, 1)), tablero.consultarCelda(new Coordenada(1, 0)) };
			// then
			assertAll("", () -> assertThat("Debería tener dos celdas contiguas.", contiguas.length, is(2)),
					() -> assertThat("Debería contener las dos celdas esperadas en cualquier orden.",
							Arrays.asList(contiguas), containsInAnyOrder(esperadas)));
		}

		/**
		 * Comprueba celdas contiguas en la esquina superior derecha del tablero.
		 */
		@DisplayName("Comprueba que se encuentran las dos celdas contiguas a celda en esquina superior derecha del tablero.")
		@Test
		void comprobarCeldasContiguasEnEsquinaSuperiorDerechaDelTablero() {
			// given
			Coordenada coordenada = new Coordenada(0, 6);
			// when
			Celda[] contiguas = tablero.consultarCeldasContiguas(coordenada); // celda consultada
			Celda[] esperadas = { // celdas contiguas
					tablero.consultarCelda(new Coordenada(0, 5)), tablero.consultarCelda(new Coordenada(1, 6)) };
			// then
			assertAll("", () -> assertThat("Debería tener dos celdas contiguas.", contiguas.length, is(2)),
					() -> assertThat("Debería contener las dos celdas esperadas en cualquier orden.",
							Arrays.asList(contiguas), containsInAnyOrder(esperadas)));
		}

		/**
		 * Comprueba celdas contiguas en la esquina inferior izquierda del tablero.
		 */
		@DisplayName("Comprueba que se encuentran las dos celdas contiguas a celda en esquina inferior izquierda del tablero.")
		@Test
		void comprobarCeldasContiguasEnEsquinaInferiorIzquierdaDelTablero() {
			// given
			Coordenada coordenada = new Coordenada(6, 0);
			// when
			Celda[] contiguas = tablero.consultarCeldasContiguas(coordenada); // celda consultada
			Celda[] esperadas = { // celdas contiguas
					tablero.consultarCelda(new Coordenada(5, 0)), tablero.consultarCelda(new Coordenada(6, 1)) };
			// then
			assertAll("", () -> assertThat("Debería tener dos celdas contiguas.", contiguas.length, is(2)),
					() -> assertThat("Debería contener las dos celdas esperadas en cualquier orden.",
							Arrays.asList(contiguas), containsInAnyOrder(esperadas)));
		}

		/**
		 * Comprueba celdas contiguas en la esquina inferior derecha del tablero.
		 */
		@DisplayName("Comprueba que se encuentran las dos celdas contiguas a celda en esquina inferior derecha del tablero.")
		@Test
		void comprobarCeldasContiguasEnEsquinaInferiorDerechaDelTablero() {
			// given
			Coordenada coordenada = new Coordenada(6, 6);
			// when
			Celda[] contiguas = tablero.consultarCeldasContiguas(coordenada); // celda consultada
			Celda[] esperadas = { // celdas contiguas
					tablero.consultarCelda(new Coordenada(5, 6)), tablero.consultarCelda(new Coordenada(6, 5)) };
			// then
			assertAll("", () -> assertThat("Debería tener dos celdas contiguas.", contiguas.length, is(2)),
					() -> assertThat("Debería contener las dos celdas esperadas en cualquier orden.",
							Arrays.asList(contiguas), containsInAnyOrder(esperadas)));
		}

		/**
		 * Comprueba celdas contiguas en la mitad del borde superior del tablero.
		 */
		@DisplayName("Comprueba que se encuentran las tres celdas contiguas a celda en mitad del borde superior del tablero.")
		@Test
		void comprobarCeldasContiguasEnMitadDelBordeSuperiorDelTablero() {
			// given
			Coordenada coordenada = new Coordenada(0, 3);
			// when
			Celda[] contiguas = tablero.consultarCeldasContiguas(coordenada); // celda consultada
			Celda[] esperadas = { // celdas contiguas
					tablero.consultarCelda(new Coordenada(0, 2)), tablero.consultarCelda(new Coordenada(0, 4)),
					tablero.consultarCelda(new Coordenada(1, 3)) };
			// then
			assertAll("", () -> assertThat("Debería tener tres celdas contiguas.", contiguas.length, is(3)),
					() -> assertThat("Debería contener las tres celdas esperadas en cualquier orden.",
							Arrays.asList(contiguas), containsInAnyOrder(esperadas)));
		}

		/**
		 * Comprueba celdas contiguas en la mitad del borde inferior del tablero.
		 */
		@DisplayName("Comprueba que se encuentran las tres celdas contiguas a celda en mitad del borde inferior del tablero.")
		@Test
		void comprobarCeldasContiguasEnMitadDelBordeInferiorDelTablero() {
			// given
			Coordenada coordenada = new Coordenada(6, 3);
			// when
			Celda[] contiguas = tablero.consultarCeldasContiguas(coordenada); // celda consultada
			Celda[] esperadas = { // celdas contiguas
					tablero.consultarCelda(new Coordenada(6, 2)), tablero.consultarCelda(new Coordenada(6, 4)),
					tablero.consultarCelda(new Coordenada(5, 3)) };
			// then
			assertAll("", () -> assertThat("Debería tener tres celdas contiguas.", contiguas.length, is(3)),
					() -> assertThat("Debería contener las tres celdas esperadas en cualquier orden.",
							Arrays.asList(contiguas), containsInAnyOrder(esperadas)));
		}

		/**
		 * Comprueba celdas contiguas en la mitad del borde izquierdo del tablero.
		 */
		@DisplayName("Comprueba que se encuentran las tres celdas contiguas a celda en mitad del borde izquierdo del tablero.")
		@Test
		void comprobarCeldasContiguasEnMitadDelBordeIzquierdoDelTablero() {
			// given
			Coordenada coordenada = new Coordenada(3, 0);
			// when
			Celda[] contiguas = tablero.consultarCeldasContiguas(coordenada); // celda consultada
			Celda[] esperadas = { // celdas contiguas
					tablero.consultarCelda(new Coordenada(2, 0)), tablero.consultarCelda(new Coordenada(4, 0)),
					tablero.consultarCelda(new Coordenada(3, 1)) };
			// then
			assertAll("", () -> assertThat("Debería tener tres celdas contiguas.", contiguas.length, is(3)),
					() -> assertThat("Debería contener las tres celdas esperadas en cualquier orden.",
							Arrays.asList(contiguas), containsInAnyOrder(esperadas)));
		}

		/**
		 * Comprueba celdas contiguas en la mitad del borde derecho del tablero.
		 */
		@DisplayName("Comprueba que se encuentran las tres celdas contiguas a celda en mitad del borde derecho del tablero.")
		@Test
		void comprobarCeldasContiguasEnMitadDelBordeDerechoDelTablero() {
			// given
			Coordenada coordenada = new Coordenada(3, 6);
			// when
			Celda[] contiguas = tablero.consultarCeldasContiguas(coordenada); // celda consultada
			Celda[] esperadas = { // celdas contiguas
					tablero.consultarCelda(new Coordenada(2, 6)), tablero.consultarCelda(new Coordenada(4, 6)),
					tablero.consultarCelda(new Coordenada(3, 5)) };
			// then
			assertAll("", () -> assertThat("Debería tener tres celdas contiguas.", contiguas.length, is(3)),
					() -> assertThat("Debería contener las tres celdas esperadas en cualquier orden.",
							Arrays.asList(contiguas), containsInAnyOrder(esperadas)));
		}
	}

	/**
	 * Colocación y consulta del número de piezas del tablero.
	 * 
	 * @see brandubh.modelo.Tablero#colocar(Pieza, Coordenada)
	 * @see brandubh.modelo.Tablero#consultarNumeroPiezas(TipoPieza)
	 */
	@Nested
	@DisplayName("Tests de colocación de piezas en el tablero.")
	@Order(4)
	class ColocarPiezasEnTablero {
		
		/**
		 * Comprueba la colocación de una única pieza el tablero.
		 * 
		 * @param fila fila
		 * @param columna columna 
		 */
		@DisplayName("Coloca correctamente una pieza.")
		@ParameterizedTest
		@MethodSource("brandubh.modelo.TestUtil#proveerCoordenadas")
		void comprobarColocarUnaPiezaEnElTablero(int fila, int columna) {
			// given tablero...
			Coordenada coordenada = new Coordenada(fila, columna);
			Pieza pieza = new Pieza(TipoPieza.ATACANTE);
			// no debería afectar en el test el tipo de pieza colocada...
			
			// when
			tablero.colocar(pieza, coordenada);

			// then
			Celda celda = tablero.obtenerCelda(coordenada);			
			assertAll("comprobación de de pieza colocada",
					() -> assertThat("Pieza mal asignada.", celda.consultarPieza(), is(pieza)),
					() -> assertFalse("La celda está vacía", celda.estaVacia()),
					() -> assertThat("Debería tener un atacante.", tablero.consultarNumeroPiezas(TipoPieza.ATACANTE),
							is(1)),
					() -> assertThat("Debería tener cero defensores.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(0)),
					() -> assertThat("Debería tener cero reyes.", tablero.consultarNumeroPiezas(TipoPieza.REY), is(0)));
		}

		/**
		 * Rellena el tablero de distintos tipos de piezas hasta ver que está completo.
		 */
		@DisplayName("Comprueba el rellenado del tablero de piezas hasta completarlo")
		@Test
		void comprobarRellenadoDelTableroConPiezas() {

			// En cada iteración lo completamos con tipos de piezas distintas
			for (TipoPieza tipoPieza : TipoPieza.values()) {
				Pieza pieza = new Pieza(tipoPieza);
				for (int ii = 0; ii < tablero.consultarNumeroFilas(); ii++) {
					for (int jj = 0; jj < tablero.consultarNumeroColumnas(); jj++) {
						Coordenada coordenada = new Coordenada(ii, jj);
						tablero.colocar(pieza, coordenada);
						Celda celda = tablero.obtenerCelda(coordenada);
						assertThat("Pieza mal asignada.", celda.consultarPieza(), is(pieza));
						assertFalse("La celda está vacía", celda.estaVacia());
					}
				}
				assertThat("Número de piezas incorrecto para tipo de pieza " + tipoPieza,
						tablero.consultarNumeroPiezas(tipoPieza), is(TOTAL_CELDAS_BRANDUBH));
			}
		}
	}	

	/**
	 * Eliminación de piezas.
	 * 
	 * @see brandubh.modelo.Tablero#eliminarPieza(Coordenada)
	 */
	@Nested
	@DisplayName("Tests de eliminación de piezas.")
	@Order(5)
	class Eliminar {

		/**
		 * Comprueba la eliminaciónde una única pieza el tablero.
		 * 
		 * @param fila fila
		 * @param columna columna
		 */
		@DisplayName("Elimina correctamente una pieza.")
		@ParameterizedTest
		@MethodSource("brandubh.modelo.TestUtil#proveerCoordenadas")
		void comprobarEliminacionUnaPiezaEnElTablero(int fila, int columna) {
			// given tablero...
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(fila, columna)); // no debería afectar en el
																							// test el tipo de pieza
																							// colocada...

			assertAll("conteo de pieza colocada",
					() -> assertThat("Debería tener un atacante.", tablero.consultarNumeroPiezas(TipoPieza.ATACANTE),
							is(1)),
					() -> assertThat("Debería tener cero defensores.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(0)),
					() -> assertThat("Debería tener cero reyes.", tablero.consultarNumeroPiezas(TipoPieza.REY), is(0)));

			// when eliminamos la pieza recién colocada...
			tablero.eliminarPieza(new Coordenada(fila, columna));

			// then
			assertAll("conteo de piezas tras eliminación",
					() -> assertThat("Debería tener cero atacantes.", tablero.consultarNumeroPiezas(TipoPieza.ATACANTE),
							is(0)),
					() -> assertThat("Debería tener cero defensores.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(0)),
					() -> assertThat("Debería tener cero reyes.", tablero.consultarNumeroPiezas(TipoPieza.REY), is(0)));

		}

		/**
		 * Comprueba la eliminación de varias piezas en el tablero.
		 * 
		 * @see brandubh.modelo.TableroTest#colocarNuevePiezasEnAspa()
		 */
		@DisplayName("Elimina correctamente varias piezas.")
		@Test
		void comprobarEliminacionDePiezasEnElTablero() {
			// given tablero...
			colocarNuevePiezasEnAspa(); // usamo el método auxiliar para colocar las piezas en aspa

			assertAll("conteo de piezas colocadas",
					() -> assertThat("Debería tener cuatro atacantes.",
							tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(4)),
					() -> assertThat("Debería tener cuatro defensores.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
					() -> assertThat("Debería tener un rey.", tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)));

			// when eliminamos TODAS las piezas previamente colocadas
			tablero.eliminarPieza(new Coordenada(0, 0));
			tablero.eliminarPieza(new Coordenada(0, 6));
			tablero.eliminarPieza(new Coordenada(1, 1));
			tablero.eliminarPieza(new Coordenada(1, 5));
			tablero.eliminarPieza(new Coordenada(3, 3));
			tablero.eliminarPieza(new Coordenada(5, 1));
			tablero.eliminarPieza(new Coordenada(5, 5));
			tablero.eliminarPieza(new Coordenada(6, 0));
			tablero.eliminarPieza(new Coordenada(6, 6));

			// then
			assertAll("conteo de piezas tras eliminación",
					() -> assertThat("Debería tener cero atacantes.", tablero.consultarNumeroPiezas(TipoPieza.ATACANTE),
							is(0)),
					() -> assertThat("Debería tener cero defensores.",
							tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(0)),
					() -> assertThat("Debería tener cero reyes.", tablero.consultarNumeroPiezas(TipoPieza.REY), is(0)));

		}
	}

	/**
	 * Clonación.
	 */
	@Nested
	@DisplayName("Tests de clonación.")
	@Order(6)
	class Clonacion {

		/**
		 * Comprueba que la clonación de un tablero vacío es correcta.
		 */
		@DisplayName("Comprueba la correcta clonación de un tablero vacío.")
		@Test
		void comprobarClonacionDeUnTableroVacio() {
			Tablero tableroClonado = tablero.clonar();
			assertAll("clonación de un tablero vacío",
					() -> assertNotSame(tableroClonado, tablero,
							"Ambas referencias apuntan al mismo objeto, no se ha clonado correctamente el tablero."),
					() -> assertEquals(tableroClonado, tablero,
							"Ambos tableros no coinciden en contenido (revisar además implementación de equals y hashCode)."));
		}

		/**
		 * Comprueba la clonación de un tablero con algunas jugadas es correcta.
		 * 
		 * @see brandubh.modelo.TableroTest#colocarNuevePiezasEnAspa()
		 */
		@DisplayName("Comprueba la correcta clonación de un tablero con algunas piezas colocadas.")
		@Test
		void comprobarClonacionDeUnTableroConAlgunosPiezas() {
			// given tablero con siete piezas colocadas en la diagonal...
			colocarNuevePiezasEnAspa();
			// when clonamos
			Tablero tableroClonado = tablero.clonar();
			assertAll("clonación de un tablero con varias piezas",
					() -> assertNotSame(tableroClonado, tablero,
							"Ambas referencias apuntan al mismo objeto, no se ha clonado correctamente el tablero."),
					() -> assertEquals(tableroClonado, tablero,
							"Ambos tableros no coinciden en contenido (revisar además implementación de equals y hashCode)."));

		}

		/**
		 * Comprueba la clonación profunda de un tablero con algunas jugadas es
		 * correcta.
		 * 
		 * @see brandubh.modelo.TableroTest#colocarNuevePiezasEnAspa()
		 */
		@DisplayName("Comprueba la correcta clonación profunda de un tablero con algunas piezas.")
		@Test
		void comprobarClonacionProfundaDeUnTableroConAlgunosPiezas() {
			// given tablero con nueve piezas colocadas en aspa...
			colocarNuevePiezasEnAspa();

			// when clonamos
			Tablero tableroClonado = tablero.clonar();
			assertAll("clonación de un tablero con varias piezas",
					() -> assertNotSame(tableroClonado, tablero,
							"Ambas referencias apuntan al mismo objeto, no se ha clonado."),
					() -> assertEquals(tableroClonado, tablero,
							"Ambos tableros no coinciden en contenido (revisar además implementación de equals y hashCode)."));

			// Comprobar que las celdas dentro de cada tablero son a su vez clones...
			for (int i = 0; i < tablero.consultarNumeroFilas(); i++) {
				for (int j = 0; j < tablero.consultarNumeroColumnas(); j++) {
					Celda original = tablero.obtenerCelda(new Coordenada(i, j));
					Celda clonada = tableroClonado.obtenerCelda(new Coordenada(i, j));
					assertAll("clonación profunda de la celdas del tablero",
							() -> assertNotSame(clonada, original,
									"Ambas referencias apuntan a la misma celda, no se ha clonado."),
							() -> assertEquals(clonada, original,
									"Ambas celdas no coinciden en contenido (revisar además implementación de equals y hashCode)."));
				}
			}
		}
	}

	/**
	 * Validación de argumentos en todos aquellos métodos públicos que definen algún
	 * argumento formal en su signatura.
	 */
	@Nested
	@DisplayName("Validación de argumentos.")
	@Order(7)
	class ValidacionDeArgumentos {

		/**
		 * Comprobación de consultas básicas con argumentos no válidos.
		 * 
		 * @see brandubh.modelo.Tablero#obtenerCelda(Coordenada)
		 * @see brandubh.modelo.Tablero#consultarCelda(Coordenada)
		 * @see brandubh.modelo.Tablero#consultarNumeroPiezas(TipoPieza)
		 */
		@Nested
		@DisplayName("Validaargumentos no válidos en consultas básicas.")
		class ConsultasBasicas {

			/**
			 * Comprueba que la obtención de celda en coordenadas incorrectas retorna nulo.
			 * 
			 * @param fila    fila
			 * @param columna columna
			 * @see brandubh.modelo.Tablero#obtenerCelda(Coordenada)
			 */
			@DisplayName("Comprueba la consulta de celda en coordenadas incorrectas del tablero")
			@ParameterizedTest
			@MethodSource("brandubh.modelo.TestUtil#proveerCoordenadasIncorrectas")
			void comprobarConsultaACeldasEnPosicionesIncorrectas(int fila, int columna) {
				// given
				Coordenada coordenada = new Coordenada(fila, columna);
				// when
				Celda celda = tablero.obtenerCelda(coordenada);
				// then
				assertNull(celda, "La celda obtenida debería valer nulo para coordenadas incorrectas.");
			}

			/**
			 * Comprueba que la consulta del número de piezas con tipo de pieza nulo retorna
			 * cero.
			 * 
			 * @see brandubh.modelo.Tablero#consultarNumeroPiezas(TipoPieza)
			 */
			@DisplayName("Comprueba que al consultar el número de piezas con tipo de pieza nulo retorna cero.")
			@Test
			void comprobarQueConsultarNumeroPiezasConNuloRetornaCero() {
				// given tablero...
				colocarNuevePiezasEnAspa();
				// when
				// then
				assertThat("Debería retornar cero al consultar el numero de piezas de tipo nulo.",
						tablero.consultarNumeroPiezas(null), is(0));
			}
		}

		/**
		 * Comprobación de consulta de celdas contiguas con argumentos no válidos.
		 * 
		 * @see brandubh.modelo.Tablero#consultarCeldasContiguas(Coordenada)
		 * @see brandubh.modelo.Tablero#consultarCeldasContiguasEnHorizontal(Coordenada)
		 * @see brandubh.modelo.Tablero#consultarCeldasContiguasEnVertical(Coordenada)
		 */
		@Nested
		@DisplayName("Validación de argumentos no válidos al consultar celdas contiguas.")
		class ConsultaCeldasContiguas {

			/**
			 * Comprueba que no hay celdas contiguas para coordenadas fuera del tablero.
			 * 
			 * @param fila    fila
			 * @param columna columna
			 */
			@ParameterizedTest
			@MethodSource("brandubh.modelo.TestUtil#proveerCoordenadasIncorrectas")
			@DisplayName("Comprueba que no hay celdas contiguas para coordenadas fuera del tablero.")
			void comprobarNoHayCeldasContiguasParaCoordenadasFueraDelTablero(int fila, int columna) {
				// given
				Coordenada coordenada = new Coordenada(fila, columna);
				// when
				Celda[] contiguas = tablero.consultarCeldasContiguas(coordenada);
				// then
				assertThat("El conjunto de celdas contiguas debería ser vacío para coordenadas fuera del tablero.",
						contiguas.length, is(0));
			}

			/**
			 * Comprueba que no hay celdas contiguas en horizontal para coordenadas fuera
			 * del tablero.
			 * 
			 * @param fila    fila
			 * @param columna columna
			 */
			@ParameterizedTest
			@MethodSource("brandubh.modelo.TestUtil#proveerCoordenadasIncorrectas")
			@DisplayName("Comprueba que no hay celdas contiguas en horizontal para coordenadas fuera del tablero.")
			void comprobarNoHayCeldasContiguasEnHorizontalParaCoordenadasFueraDelTablero(int fila, int columna) {
				// given
				Coordenada coordenada = new Coordenada(fila, columna);
				// when
				Celda[] contiguas = tablero.consultarCeldasContiguasEnHorizontal(coordenada);
				// then
				assertThat(
						"El conjunto de celdas contiguas en horizontal debería ser vacío para coordenadas fuera del tablero.",
						contiguas.length, is(0));
			}

			/**
			 * Comprueba que no hay celdas contiguas en vertical para coordenadas fuera del
			 * tablero.
			 * 
			 * @param fila    fila
			 * @param columna columna
			 */
			@ParameterizedTest
			@MethodSource("brandubh.modelo.TestUtil#proveerCoordenadasIncorrectas")
			@DisplayName("Comprueba que no hay celdas contiguas en vertical para coordenadas fuera del tablero.")
			void comprobarNoHayCeldasContiguasEnVerticalParaCoordenadasFueraDelTablero(int fila, int columna) {
				// given
				Coordenada coordenada = new Coordenada(fila, columna);
				// when
				Celda[] contiguas = tablero.consultarCeldasContiguasEnVertical(coordenada);
				// then
				assertThat(
						"El conjunto de celdas contiguas en vertical debería ser vacío para coordenadas fuera del tablero.",
						contiguas.length, is(0));
			}

		}

		/**
		 * Comprobación de colocar con argumentos no válidos.
		 * 
		 * @see brandubh.modelo.Tablero#colocar(Pieza, Coordenada)
		 */
		@Nested
		@DisplayName("Validación de argumentos no válidos al colocar.")
		class Colocar {

			/**
			 * Comprueba que se ignora colocar si se invoca con una pieza de valor nulo.
			 * 
			 * @param fila    fila
			 * @param columna columna
			 */
			@ParameterizedTest
			@MethodSource("brandubh.modelo.TestUtil#proveerCoordenadas")
			@DisplayName("Comprueba que ignora colocar los valores nulos de pieza.")
			void comprobarIgnoraColocarConPiezaNula(int fila, int columna) {
				Coordenada coordenada = new Coordenada(fila, columna);
				Pieza pieza = new Pieza(TipoPieza.ATACANTE); // para el test no debería afectar el tipo de pieza
				tablero.colocar(pieza, coordenada);
				assertThat("La celda no contiene la pieza añadida.", tablero.obtenerCelda(coordenada).consultarPieza(),
						is(pieza));
				tablero.colocar(null, coordenada);
				assertThat("Debería seguir estando la pieza añadida anteriormente.",
						tablero.obtenerCelda(coordenada).consultarPieza(), is(pieza));
			}

			/**
			 * Comprueba que se ignora colocar si se invoca con una coordenada nula.
			 * 
			 * @param fila    fila
			 * @param columna columna
			 */
			@ParameterizedTest
			@MethodSource("brandubh.modelo.TestUtil#proveerCoordenadas")
			@DisplayName("Comprueba que ignora colocar con valores nulos de pieza.")
			void comprobarIgnoraColocarConCoordenadaNula(int fila, int columna) {
				Coordenada coordenada = new Coordenada(fila, columna);
				Pieza pieza = new Pieza(TipoPieza.ATACANTE); // para el test no debería afectar el tipo de pieza
				tablero.colocar(pieza, coordenada);
				assertThat("La celda no contiene la pieza añadida.", tablero.obtenerCelda(coordenada).consultarPieza(),
						is(pieza));
				tablero.colocar(new Pieza(TipoPieza.DEFENSOR), null);
				assertThat("Debería seguir estando la pieza añadida anteriormente.",
						tablero.obtenerCelda(coordenada).consultarPieza(), is(pieza));
			}

			/**
			 * Comprueba que no coloca piezas atacantes en coordenadas fuera del tablero.
			 * 
			 * @param fila    fila
			 * @param columna columna
			 */
			@DisplayName("Comprueba que no deja colocar piezas atacantes en coordenada incorrectas fuera del tablero")
			@ParameterizedTest
			@MethodSource("brandubh.modelo.TestUtil#proveerCoordenadasIncorrectas")
			void comprobarColocarPiezasAtacantesEnCeldasIncorrectas(int fila, int columna) {
				tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(fila, columna));
				assertThat("No debería colocar una pieza atacante en una coordenada fuera del tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(0));
			}

			/**
			 * Comprueba que no coloca piezas defensoras en coordenadas fuera del tablero.
			 * 
			 * @param fila    fila
			 * @param columna columna
			 */
			@DisplayName("Comprueba que no deja colocar piezas defensoras en coordenada incorrectas fuera del tablero")
			@ParameterizedTest
			@MethodSource("brandubh.modelo.TestUtil#proveerCoordenadasIncorrectas")
			void comprobarColocarPiezasDefensorasEnCoordenadasIncorrectas(int fila, int columna) {
				tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(fila, columna));
				assertThat("No debería colocar una pieza defensora en una coordenada fuera del tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(0));
			}

			/**
			 * Comprueba que no coloca piezas de tipo rey en coordenadas fuera del tablero.
			 * 
			 * @param fila    fila
			 * @param columna columna
			 */
			@DisplayName("Comprueba que no deja colocar piezas de tipo rey en coordenada incorrectas fuera del tablero")
			@ParameterizedTest
			@MethodSource("brandubh.modelo.TestUtil#proveerCoordenadasIncorrectas")
			void comprobarColocarReyEnCoordenadasIncorrectas(int fila, int columna) {
				tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(fila, columna));
				assertThat("No debería colocar un rey en una coordenada fuera del tablero.",
						tablero.consultarNumeroPiezas(TipoPieza.REY), is(0));
			}
		}

		/**
		 * Comprobación de eliminar con argumentos no válidos.
		 * 
		 * @see brandubh.modelo.Tablero#eliminarPieza(Coordenada)
		 */
		@Nested
		@DisplayName("Validación de argumentos no válidos al eliminar.")
		class Eliminar {

			/**
			 * Comprueba que la eliminación de piezas en el tablero no hace nada ante
			 * coordenadas nulas.
			 */
			@DisplayName("Comprueba que al eliminar con coordenada nula no tiene efecto.")
			@Test
			void comprobarQueEliminarConCoordenadaNulaNoTieneEfecto() {
				// given tablero...
				colocarNuevePiezasEnAspa();
				// when intentamos eliminar con coordenada valor nulo
				tablero.eliminarPieza(null);
				// then el número de piezas no cambia
				assertAll("conteo de piezas tras eliminación",
						() -> assertThat("Debería tener cuatro atacantes al eliminar con coordenada nula.",
								tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(4)),
						() -> assertThat("Debería tener cuatro defensores al eliminar con coordenada nula.",
								tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
						() -> assertThat("Debería tener cero reyes al eliminar con coordenada nula.",
								tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)));

			}
			
			/**
			 * Comprueba que la eliminación de piezas en el tablero no hace nada ante
			 * coordenadas incorrectas.
			 * 
			 * @param fila fila
			 * @param columna columna
			 */
			@DisplayName("Comprueba que al eliminar con coordenada nula no tiene efecto.")
			@ParameterizedTest
			@MethodSource("brandubh.modelo.TestUtil#proveerCoordenadasIncorrectas")
			void comprobarQueEliminarConCoordenadaIncorrectaNoTieneEfecto(int fila, int columna) {
				// given tablero...
				colocarNuevePiezasEnAspa();
				// when intentamos eliminar con coordenada valor nulo
				tablero.eliminarPieza(new Coordenada(fila, columna));
				// then el número de piezas no cambia
				assertAll("conteo de piezas tras eliminación",
						() -> assertThat("Debería tener cuatro atacantes al eliminar con coordenada nula.",
								tablero.consultarNumeroPiezas(TipoPieza.ATACANTE), is(4)),
						() -> assertThat("Debería tener cuatro defensores al eliminar con coordenada nula.",
								tablero.consultarNumeroPiezas(TipoPieza.DEFENSOR), is(4)),
						() -> assertThat("Debería tener cero reyes al eliminar con coordenada nula.",
								tablero.consultarNumeroPiezas(TipoPieza.REY), is(1)));

			}
		}

	}

	/**
	 * Tests de conversión a texto.
	 * 
	 * @see brandubh.modelo.Tablero#aTexto()
	 */
	@Nested
	@DisplayName("Conversión a texto.")
	@Order(8)
	class ConversionATexto {

		/** Genera la cadena de texto correcta para un tablero vacío. */
		@DisplayName("Comprueba la generación de la cadena de texto con tablero vacío")
		@Test
		void comprobarCadenaTextoConTableroVacio() {
			String cadenaEsperada = """
					7 - - - - - - -
					6 - - - - - - -
					5 - - - - - - -
					4 - - - - - - -
					3 - - - - - - -
					2 - - - - - - -
					1 - - - - - - -
					  a b c d e f g""";
			cadenaEsperada = cadenaEsperada.replaceAll("\\s", "");
			// eliminamos espacios/tabuladores para comparar
			String salida = tablero.aTexto().replaceAll("\\s", "");
			assertEquals(cadenaEsperada, salida, "La cadena de texto generada para un tablero vacío es incorecta.");
		}

		/**
		 * Genera la cadena de texto correcta para un tablero con algunos atacantes
		 * colocados en las esquinas del tablero.
		 */
		@DisplayName("Comprueba la generación de la cadena de texto con tablero con algunos atacantes en las esquinas")
		@Test
		void comprobarCadenaTextoConTableroConAtacantesEnEsquinas() {
			String cadenaEsperada = """
					7 A - - - - - A
					6 - - - - - - -
					5 - - - - - - -
					4 - - - - - - -
					3 - - - - - - -
					2 - - - - - - -
					1 A - - - - - A
					  a b c d e f g""";
			cadenaEsperada = cadenaEsperada.replaceAll("\\s", "");
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(0, 0));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(0, 6));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(6, 0));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(6, 6));
			// eliminamos espacios/tabuladores para comparar
			String salida = tablero.aTexto().replaceAll("\\s", "");
			assertEquals(cadenaEsperada, salida,
					"La cadena de texto generada para un tablero con atacantes en las esquinas es incorrecta.");
		}

		/**
		 * Genera la cadena de texto correcta para un tablero con algunos atacantes,
		 * defensores y un rey.
		 */
		@DisplayName("Comprueba la generación de la cadena de texto con algunos atacantes, defensores y un rey.")
		@Test
		void comprobarCadenaTextoConTableroConPeonesYDamas() {
			String cadenaEsperada = """
					7 A - - - - - A
					6 - D - - - D -
					5 - - - - - - -
					4 - - - R - - -
					3 - - - - - - -
					2 - D - - - D -
					1 A - - - - - A
					  a b c d e f g""";
			cadenaEsperada = cadenaEsperada.replaceAll("\\s", "");
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(0, 0));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(0, 6));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(6, 0));
			tablero.colocar(new Pieza(TipoPieza.ATACANTE), new Coordenada(6, 6));

			tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(1, 1));
			tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(1, 5));
			tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(5, 1));
			tablero.colocar(new Pieza(TipoPieza.DEFENSOR), new Coordenada(5, 5));

			tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(3, 3));

			// eliminamos espacios/tabuladores para comparar
			String salida = tablero.aTexto().replaceAll("\\s", "");
			assertEquals(cadenaEsperada, salida,
					"La cadena de texto generada para un tablero con atacantes, defensores y rey en aspa es incorrecta.");
		}
	}
}
