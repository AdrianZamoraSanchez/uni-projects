package tafl;


import org.junit.platform.suite.api.SelectPackages;
import org.junit.platform.suite.api.Suite;
import org.junit.platform.suite.api.SuiteDisplayName;

/**
 * Suite ejecutando los tests de nivel 9 de la práctica Tafl-1.0 (ver README.txt).
 * Equivalente a ejecutar {@link tafl.SuiteAllTests} con todos lo tests.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@SelectPackages({	
	"tafl.control.ardri.basico",
	"tafl.control.ardri.medio",
	"tafl.control.ardri.avanzado",
	"tafl.control.brandubh.basico",
	"tafl.control.brandubh.medio",
	"tafl.control.brandubh.avanzado",
	"tafl.excepcion",
	"tafl.modelo",
	"tafl.util"})

@Suite
@SuiteDisplayName("Tests de paquetes control (completo), excepcion, modelo y util completos.")
public class SuiteLevel9Tests {

}
