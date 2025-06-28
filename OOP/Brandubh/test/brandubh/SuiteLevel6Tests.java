package brandubh;


import org.junit.platform.suite.api.ExcludePackages;
import org.junit.platform.suite.api.SelectPackages;
import org.junit.platform.suite.api.Suite;
import org.junit.platform.suite.api.SuiteDisplayName;

/**
 * Suite ejecutando los tests de nivel 6 de la práctica Brandubh-1.0 (ver README.txt).
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@SelectPackages({
	"brandubh.control",
	"brandubh.modelo",
	"brandubh.util"})
@ExcludePackages({"brandubh.control.avanzado", "brandubh.control.medio"})
@Suite
@SuiteDisplayName("Tests de paquetes control (tests basicos), modelo y util completos.")
public class SuiteLevel6Tests {

}
