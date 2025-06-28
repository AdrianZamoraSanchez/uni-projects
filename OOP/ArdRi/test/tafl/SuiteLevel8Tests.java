package tafl;


import org.junit.platform.suite.api.ExcludePackages;
import org.junit.platform.suite.api.SelectPackages;
import org.junit.platform.suite.api.Suite;
import org.junit.platform.suite.api.SuiteDisplayName;

/**
 * Suite ejecutando los tests de nivel 8 de la práctica Tafl-1.0 (ver README.txt).
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@SelectPackages({
	"tafl.control.ardri.basico",
	"tafl.control.ardri.medio",
	"tafl.control.brandubh.basico",
	"tafl.control.brandubh.medio",
	"tafl.control",
	"tafl.excepcion",
	"tafl.modelo",
	"tafl.util"})
@ExcludePackages({"tafl.control.avanzado"})
@Suite
@SuiteDisplayName("Tests de paquetes control (tests basicos y medios), modelo y util completos.")
public class SuiteLevel8Tests {

}
