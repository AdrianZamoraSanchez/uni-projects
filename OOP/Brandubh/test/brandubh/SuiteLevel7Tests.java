package brandubh;


import org.junit.platform.suite.api.ExcludePackages;
import org.junit.platform.suite.api.SelectPackages;
import org.junit.platform.suite.api.Suite;
import org.junit.platform.suite.api.SuiteDisplayName;

/**
 * Suite ejecutando los tests de nivel 7 de la práctica Brandubh-1.0 (ver README.txt).
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@SelectPackages({
	"brandubh.control",
	"brandubh.modelo",
	"brandubh.util"})
@ExcludePackages({"brandubh.control.avanzado"})
@Suite
@SuiteDisplayName("Tests de paquetes control (tests basicos y medios), modelo y util completos.")
public class SuiteLevel7Tests {

}
