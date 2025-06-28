package brandubh.util;
import java.util.Arrays;
import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * Tests sobre el tipo de celda.
 * 
 * La enumeracion es un elemento básico que debería ser implementado y probado
 * en primer lugar antes de proseguir con el resto de clases.
 * 
 * @author <a href="mailto:rmartico@ubu.es">Raúl Marticorena Sánchez</a>
 * @version 1.0 20230706
 * 
 */
@DisplayName("Tests sobre tipos de celda (sin dependencia de otras clases).")
@Timeout(value = 1, unit = TimeUnit.SECONDS) // Time out global para todos los tests salvo los de ciclo de vida
public class TipoCeldaTest {

	/**
	 * Comprobar el correcto número de valores.
	 */
	@DisplayName("Comprueba el número de valores definidos.")
	@Test
	public void probarNumeroValores() {
		assertThat("La enumeración debe tener exactamente TRES valores definidos (no importa el orden).",
				TipoCelda.values().length, is(3));
	}

	/**
	 * Comprueba valores correctos definidos.
	 */
	@DisplayName("Comprueba los valores definidos.")
	@Test
	public void probarValores() {		
		 assertThat("El tipo enumerado no contiene todos los valores esperados (no importa el orden).",
				 Arrays.asList(TipoCelda.values()), containsInAnyOrder(TipoCelda.NORMAL, TipoCelda.PROVINCIA, 
				 TipoCelda.TRONO));
	}
}
