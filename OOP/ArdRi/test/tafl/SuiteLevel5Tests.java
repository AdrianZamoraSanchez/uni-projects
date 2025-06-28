package tafl;


import org.junit.platform.suite.api.ExcludeClassNamePatterns;
import org.junit.platform.suite.api.SelectPackages;
import org.junit.platform.suite.api.Suite;
import org.junit.platform.suite.api.SuiteDisplayName;

/**
 * Suite ejecutando los tests de nivel 5 de la práctica Tafl-1.0 (ver README.txt).
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@SelectPackages({
	"tafl.excepcion",
	"tafl.modelo",
	"tafl.util"})
@ExcludeClassNamePatterns({"^.*TableroTest?$"})
@Suite
@SuiteDisplayName("Tests unitarios y de integración de paquetes excepcion, modelo y util a excepción de Tablero.")
public class SuiteLevel5Tests {

}
