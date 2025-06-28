package brandubh;


import org.junit.platform.suite.api.ExcludeClassNamePatterns;
import org.junit.platform.suite.api.SelectPackages;
import org.junit.platform.suite.api.Suite;
import org.junit.platform.suite.api.SuiteDisplayName;

/**
 * Suite ejecutando los tests de nivel 4 de la práctica Brandubh-1.0 (ver README.txt).
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@SelectPackages({
	"brandubh.modelo",
	"brandubh.util"})
@ExcludeClassNamePatterns({"^.*TableroTest?$"})
@Suite
@SuiteDisplayName("Tests unitarios y de integración de paquetes modelo y util a excepción de Tablero.")
public class SuiteLevel4Tests {

}
