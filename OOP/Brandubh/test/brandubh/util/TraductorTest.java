package brandubh.util;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Tests sobre el traductor.
 * 
 * @author <a href="mailto:rmartico@ubu.es">Raúl Marticorena Sánchez</a>
 * @version 1.0 20230706
 * @see brandubh.util.Coordenada
 */
@DisplayName("Tests sobre la traducción a notación algebraica (depende de Coordenada).")
@Timeout(value = 1, unit = TimeUnit.SECONDS) // Time out global para todos los tests salvo los de ciclo de vida
public class TraductorTest {
	
	/**
	 * Comprueba la conversión de formato texto en notación algebraica a coordenada
	 * con valores correctos.
	 */
	@DisplayName("Comprueba la conversión de texto en notación algebraica a coordenada con valores correctos.")
	@Test
	void comprobarConversionTextoACoordenadaConValoresCorrectos() {
		assertAll("conversiones a texto con valores correctos de coordenada",
				() -> assertThat("Conversión incorrecta para a7.",
						Traductor.consultarCoordenadaParaNotacionAlgebraica("a7"), is(new Coordenada(0, 0))),
				() -> assertThat("Conversión incorrecta para a1.",
						Traductor.consultarCoordenadaParaNotacionAlgebraica("a1"), is(new Coordenada(6, 0))),
				() -> assertThat("Conversión incorrecta para g7.",
						Traductor.consultarCoordenadaParaNotacionAlgebraica("g7"), is(new Coordenada(0, 6))),
				() -> assertThat("Conversión incorrecta para g1.",
						Traductor.consultarCoordenadaParaNotacionAlgebraica("g1"), is(new Coordenada(6, 6))),
				() -> assertThat("Conversión incorrecta para e4.",
						Traductor.consultarCoordenadaParaNotacionAlgebraica("e4"), is(new Coordenada(3, 4))),
				() -> assertThat("Conversión incorrecta para d5.",
						Traductor.consultarCoordenadaParaNotacionAlgebraica("d5"), is(new Coordenada(2, 3))));
	}

	/**
	 * Comprueba la conversión de formato texto en notación algebraica a coordenada
	 * con valores incorrectos. Debería devolver siempre null.
	 */
	@DisplayName("Comprueba la conversión de texto en notación algebraica a coordenada con valores incorrectos.")
	@Test
	void comprobarConversionTextoACoordenadaConValoresIncorrectos() {
		assertAll("conversiones a texto con valores incorrectos de coordenada",
				() -> assertNull(Traductor.consultarCoordenadaParaNotacionAlgebraica("g8"),
						"Traducción incorrecta para coordenada inválida g8, debería ser null."),
				() -> assertNull(Traductor.consultarCoordenadaParaNotacionAlgebraica("a0"),
						"Traducción incorrecta para coordenada inválida a0, debería ser null."),
				() -> assertNull(Traductor.consultarCoordenadaParaNotacionAlgebraica("i15"),
						"Traducción incorrecta para coordenada inválida i15, debería ser null."),
				() -> assertNull(Traductor.consultarCoordenadaParaNotacionAlgebraica("a9"),
						"Traducción incorrecta para coordenada inválida a9, debería ser null."),
				() -> assertNull(Traductor.consultarCoordenadaParaNotacionAlgebraica("x9"),
						"Traducción incorrecta para coordenada inválida x9, debería ser null."));
	}

	/**
	 * Comprueba la conversión de coordenadas correctas a notación algebraica.
	 */
	@DisplayName("Comprueba la conversión de coordenadas correctas a notación algebraica.")
	@Test
	void comprobarConversionCoordenadaATextoConValoresCorrectos() {
		assertAll("conversiones de coordenada correcta a texto",
				() -> assertThat("Conversión incorrecta para (0,0).",
						Traductor.consultarTextoEnNotacionAlgebraica(new Coordenada(0, 0)), is("a7")),
				() -> assertThat("Conversión incorrecta para (0,6).",
						Traductor.consultarTextoEnNotacionAlgebraica(new Coordenada(0, 6)), is("g7")),
				() -> assertThat("Conversión incorrecta para (6,0).",
						Traductor.consultarTextoEnNotacionAlgebraica(new Coordenada(6, 0)), is("a1")),
				() -> assertThat("Conversión incorrecta para (6,6).",
						Traductor.consultarTextoEnNotacionAlgebraica(new Coordenada(6, 6)), is("g1")),
				() -> assertThat("Conversión incorrecta para (3,2).",
						Traductor.consultarTextoEnNotacionAlgebraica(new Coordenada(3, 2)), is("c4")),
				() -> assertThat("Conversión incorrecta para (4,5).",
						Traductor.consultarTextoEnNotacionAlgebraica(new Coordenada(4, 5)), is("f3")));
	}

	/**
	 * Comprueba la conversión de coordenadas incorrectas a formato texto. Debería
	 * devolver siempre null.
	 */
	@DisplayName("Comprueba la conversión de coordenadas incorrectas a notación algebraica.")
	@Test
	void comprobarConversionCoordenadaATextoConValoresIncorrectos() {
		assertAll("Conversión de coordenadas incorrectas a notación algebraica.",
				() -> assertNull(Traductor.consultarTextoEnNotacionAlgebraica(new Coordenada(-1, 0)),
						"Traducción incorrecta para coordenada inválida (-1,0), debería ser null."),
				() -> assertNull(Traductor.consultarTextoEnNotacionAlgebraica(new Coordenada(7, 0)),
						"Traducción incorrecta para coordenada inválida (7,0), debería ser null."),
				() -> assertNull(Traductor.consultarTextoEnNotacionAlgebraica(new Coordenada(0, 7)),
						"Traducción incorrecta para coordenada inválida (0,7), debería ser null."),
				() -> assertNull(Traductor.consultarTextoEnNotacionAlgebraica(new Coordenada(7, 7)),
						"Traducción incorrecta para coordenada inválida (7,7), debería ser null."),
				() -> assertNull(Traductor.consultarTextoEnNotacionAlgebraica(new Coordenada(100, 100)),
						"Traducción incorrecta para coordenada inválida (100,100), debería ser null."));
	}
	
	/**
	 * Comprueba el correcto chequeo de formato texto en notación algebraica a coordenada
	 * con valores correctos.
	 * 
	 * @param texto texto con la coordenada en notación algebraica
	 */
	@DisplayName("Comprueba el chequeo de texto en notación algebraica a coordenada con valores correctos.")
	@ParameterizedTest
	@CsvSource({"a7", "a1", "g7", "b2", "g1", "e4", "d5", "c4", "e6", "f3", "f1"})
	void comprobarChequeoDeTextoACoordenadaConValoresCorrectos(String texto) {
		assertThat("Chequeo incorrecto para texto válido.",
						Traductor.esTextoCorrectoParaCoordenada(texto), is(true));
	}
	
	
	/**
	 * Comprueba el chequeo de formato texto en notación algebraica a coordenada
	 * con valores incorrectos. Debería devolver siempre false.
	 * 
	 * @param texto texto con la coordenada en notación algebraica
	 */
	@DisplayName("Comprueba el chequeo de texto en notación algebraica a coordenada con valores incorrectos.")
	@ParameterizedTest
	@CsvSource({"g8", "8g", "0a", "1a", "a0", "i15", "a9", "6g", "x9", "A1", "1A"})
	void comprobarChequeoDeTextoACoordenadaConValoresIncorrectos(String texto) {
		assertThat("Chequeo incorrecto para texto inválido.",
						Traductor.esTextoCorrectoParaCoordenada(texto), is(false));
	}
	
	
}
