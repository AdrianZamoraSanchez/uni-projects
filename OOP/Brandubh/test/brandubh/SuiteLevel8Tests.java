package brandubh;


import org.junit.platform.suite.api.SelectPackages;
import org.junit.platform.suite.api.Suite;
import org.junit.platform.suite.api.SuiteDisplayName;

/**
 * Suite ejecutando los tests de nivel 8 de la práctica Brandubh-1.0 (ver README.txt).
 * Equivalente a ejecutar {@link brandubh.SuiteAllTests} con todos lo tests.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@SelectPackages({
	"brandubh.control",
	"brandubh.modelo",
	"brandubh.util"})

@Suite
@SuiteDisplayName("Tests de paquetes control (completo), modelo y util completos.")
public class SuiteLevel8Tests {

}
