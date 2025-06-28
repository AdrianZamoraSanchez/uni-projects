package tafl;


import org.junit.platform.suite.api.SelectPackages;
import org.junit.platform.suite.api.Suite;
import org.junit.platform.suite.api.SuiteDisplayName;

/**
 * Suite ejecutando los tests de nivel 3 de la práctica Tafl-1.0 (ver README.txt).
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.1
 */
@SelectPackages({
	"tafl.excepcion",
	"tafl.util"})
@Suite
@SuiteDisplayName("Tests unitarios de clases básicas y excepciones (con dependencias mínimas de compilación).")
public class SuiteLevel3Tests {

}
