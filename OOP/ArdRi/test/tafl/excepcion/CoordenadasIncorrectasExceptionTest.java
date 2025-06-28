package tafl.excepcion;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertAll;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Pruebas unitarias sobre la excepción CoordenadasIncorrectasException.
 * 
 * No se comprueban los constructores, sino solo la cláusula de herencia.
 * 
 * @author <a href="mailto:rmartico@ubu.es">Raúl Marticorena Sánchez</a>
 * @version 2.0 20221008
 * 
 */
@DisplayName("Tests sobre CoordenadasIncorrectasException")
public class CoordenadasIncorrectasExceptionTest {
	
	/**
	 * Correcta definición de cláusula de herencia.
	 */
	@DisplayName("Comprobar que la cláusula extends de la excepción es correcta.")
	@Test
	public void probarCorrectaHerencia() {
		// se necesitan ambos asertos por la relaciones de herencia entre ambas clases
		assertAll("incorrecta cláusula de herencia en la excepción",
			() -> assertThat("La clase CoordenadasIncorrectasException debe heredar de Exception.", 
					Exception.class.isAssignableFrom(CoordenadasIncorrectasException.class), is(true)),
			
			() -> assertThat("La clase CoordenadasIncorrectasException NO debe heredar de RuntimeException.",
					RuntimeException.class.isAssignableFrom(CoordenadasIncorrectasException.class), is(false))
			);			
	} 
}

