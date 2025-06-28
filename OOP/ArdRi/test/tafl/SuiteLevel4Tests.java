package tafl;


import org.junit.platform.suite.api.ExcludeClassNamePatterns;
import org.junit.platform.suite.api.SelectPackages;
import org.junit.platform.suite.api.Suite;
import org.junit.platform.suite.api.SuiteDisplayName;

/**
 * Suite ejecutando los tests de nivel 4 de la práctica Tafl-1.0 (ver README.txt).
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.1
 */
@SelectPackages({
	"tafl.excepcion",
	"tafl.modelo",
	"tafl.util"})
@ExcludeClassNamePatterns({"^.*TableroTest?$", "^.*CeldaTest?$", "^.*JugadaTest?$",})
@Suite
@SuiteDisplayName("Tests unitarios de excepciones, clases básicas del modelo  y util (con dependencias mínimas de compilación).")
public class SuiteLevel4Tests {

}
