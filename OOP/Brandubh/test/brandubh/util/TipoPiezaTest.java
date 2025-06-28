package brandubh.util;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertAll;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * Tests sobre el tipo de las piezas.
 * 
 * La enumeracion es un elemento básico que debería ser implementado 
 * y probado en primer lugar antes de proseguir con el resto de clases.
 * 
 * @author <a href="mailto:rmartico@ubu.es">Raúl Marticorena Sánchez</a>
 * @version 1.0 20230706
 * @see brandubh.util.Color
 * 
 */
@DisplayName("Tests sobre tipos de pieza (sin dependencia de otras clases).")
@Timeout(value = 1, unit = TimeUnit.SECONDS) // Time out global para todos los tests salvo los de ciclo de vida
public class TipoPiezaTest {
	
	/**
	 * Comprobar el correcto número de valores.
	 */
	@DisplayName("Comprueba el número de valores definidos.")
	@Test
	public void probarNumeroValores() {
		assertThat("La enumeración debe tener exactamente TRES valores definidos (no importa el orden).", TipoPieza.values().length, is(3));
	}
		
	/**
	 * Comprueba correctos textos para cada tipo de pieza.
	 */
	@DisplayName("Comprueba los textos literales para cada valor.")
	@Test
	public void probarTextos() {
		assertAll("comprobando textos correctos para cada valor del tipo enumerado ",
			() -> assertThat("Texto mal definido para un atacante.", 
					TipoPieza.ATACANTE.toChar(), is('A')),
			
			() -> assertThat("Texto mal definido para un defensor.",
					TipoPieza.DEFENSOR.toChar(), is('D')),
			
			() -> assertThat("Texto mal definido para un rey.", 
					TipoPieza.REY.toChar(), is('R'))	

			);			
	}	
	
	/**
	 * Comprueba correctos colores para cada tipo de pieza.
	 */
	@DisplayName("Comprueba los colores para cada valor.")
	@Test
	public void comprobarColor() {
		assertAll("comprobando colores correctos para cada valor del tipo enumerado ",
				() -> assertThat("Color mal definido para un atacante.", 
						TipoPieza.ATACANTE.consultarColor(), is(Color.NEGRO)),
				
				() -> assertThat("Color mal definido para un defensor.",
						TipoPieza.DEFENSOR.consultarColor(), is(Color.BLANCO)),
				
				() -> assertThat("Color mal definido para el rey.", 
						TipoPieza.REY.consultarColor(), is(Color.BLANCO))	

				);
	}
}
