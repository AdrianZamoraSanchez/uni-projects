package tafl;


import org.junit.platform.suite.api.SelectPackages;
import org.junit.platform.suite.api.Suite;
import org.junit.platform.suite.api.SuiteDisplayName;

/**
 * Suite ejecutando los tests de nivel 6 de la práctica Tafl-1.0 (ver README.txt).
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@SelectPackages({
	"tafl.excepcion",
	"tafl.modelo",
	"tafl.util"})
@Suite
@SuiteDisplayName("Tests de paquetes excepcion, modelo y util completos.")
public class SuiteLevel6Tests {

}
