package tafl.control.brandubh.medio;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Timeout.ThreadMode.SEPARATE_THREAD;
import static tafl.control.TestUtil.fabricarJugada;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import tafl.control.ArbitroBrandubh;
import tafl.excepcion.CoordenadasIncorrectasException;
import tafl.modelo.Pieza;
import tafl.modelo.Tablero;
import tafl.util.Color;
import tafl.util.Coordenada;
import tafl.util.TipoPieza;

/**
 * Comprobación del ArbitroBrandubh de victoria del defensor alcanzando provincias.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0.0.1
 */
@DisplayName("Tests del arbitro sobre victorias del defensor alcanzando provincias.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public class ArbitroVictoriaDefensorTest extends tafl.control.medio.ArbitroAbstractoVictoriaDefensorTest {

	/** Generación del árbitro para testing. */
	@BeforeEach
	void inicializar() {
		// Inyección de tablero para testing...
		tablero = new Tablero();
		arbitro = new ArbitroBrandubh(tablero);
		arbitro.cambiarTurno(); // dejamos el turno en defensor
		// y NO colocamos piezas para mover directamente el rey sobre el tablero...
	}	
	
	/**
	 * Comprueba condiciones de victoria al alcanzar el rey una provincia.
	 */
	private void comprobarReyAlcanzaProvincia() {
		assertAll("victoria alcanzando provincia",
				() -> assertThat("No debería ganar el atacante.", arbitro.haGanadoAtacante(), is(false)),
				() -> assertThat("Debería ganar el defensor.", arbitro.haGanadoRey(), is(true)),
				() -> assertThat("El turno debería ser del jugado defensor.", arbitro.consultarTurno(),
						is(Color.BLANCO)));
	}
	
	/**
	 * Comprueba que se alcanza la esquina superior izquierda en horizontal por el rey.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 */
	@Test
	@DisplayName("Comprobar que se alcanza la esquina superior izquierda en horizontal por el rey.")
	void comprobarAlcanceEsquinaSuperiorIzquierdaEnHorizontal()  throws CoordenadasIncorrectasException {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(0, 3));
		arbitro.mover(fabricarJugada(tablero, 0, 3, 0, 0));
		comprobarReyAlcanzaProvincia();
	}
	
	/**
	 * Comprueba que se alcanza la esquina superior derecha en horizontal por el rey.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 */
	@Test
	@DisplayName("Comprobar que se alcanza la esquina superior derecha en horizontal por el rey.")
	void comprobarAlcanceEsquinaSuperiorDerechaEnHorizontal()  throws CoordenadasIncorrectasException {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(0, 3));
		arbitro.mover(fabricarJugada(tablero, 0, 3, 0 , 6));
		comprobarReyAlcanzaProvincia();
	}
	
	/**
	 * Comprueba que se alcanza la esquina superior izquierda en horizontal por el rey.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 */
	@Test
	@DisplayName("Comprobar que se alcanza la esquina inferior izquierda en horizontal por el rey.")
	void comprobarAlcanceEsquinaInferiorIzquierdaEnHorizontal()  throws CoordenadasIncorrectasException {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(6, 3));
		arbitro.mover(fabricarJugada(tablero, 6, 3, 6, 0));
		comprobarReyAlcanzaProvincia();
	}
	
	/**
	 * Comprueba que se alcanza la esquina inferior derecha en horizontal por el rey.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 */
	@Test
	@DisplayName("Comprobar que se alcanza la esquina inferior derecha en horizontal por el rey.")
	void comprobarAlcanceEsquinaInferiorDerechaEnHorizontal()  throws CoordenadasIncorrectasException {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(6, 3));
		arbitro.mover(fabricarJugada(tablero, 6, 3, 6, 6));
		comprobarReyAlcanzaProvincia();
	}
	
	/**
	 * Comprueba que se alcanza la esquina superior izquierda en vertical por el rey.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 */
	@Test
	@DisplayName("Comprobar que se alcanza la esquina superior izquierda en vertical por el rey.")
	void comprobarAlcanceEsquinaSuperiorIzquierdaEnVertical()  throws CoordenadasIncorrectasException {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(3, 0));
		arbitro.mover(fabricarJugada(tablero, 3, 0, 0, 0));
		comprobarReyAlcanzaProvincia();
	}
	
	/**
	 * Comprueba que se alcanza la esquina superior derecha en vertical por el rey.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 */
	@Test
	@DisplayName("Comprobar que se alcanza la esquina superior derecha en vertical por el rey.")
	void comprobarAlcanceEsquinaSuperiorDerechaEnVertical()  throws CoordenadasIncorrectasException {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(3, 6));
		arbitro.mover(fabricarJugada(tablero, 3, 6, 0, 6));
		comprobarReyAlcanzaProvincia();
	}
	
	/**
	 * Comprueba que se alcanza la esquina inferior izquierda en vertical por el rey.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 */
	@Test
	@DisplayName("Comprobar que se alcanza la esquina inferior izquierda en vertical por el rey.")
	void comprobarAlcanceEsquinaInferiorIzquierdaEnVertical()  throws CoordenadasIncorrectasException {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(3, 0));
		arbitro.mover(fabricarJugada(tablero, 3, 0, 6, 0));
		comprobarReyAlcanzaProvincia();
	}
	
	/**
	 * Comprueba que se alcanza la esquina inferior derecha en vertical por el rey.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 */
	@Test
	@DisplayName("Comprobar que se alcanza la esquina superior derecha en vertical por el rey.")
	void comprobarAlcanceEsquinaInferiorDerechaEnVertical()  throws CoordenadasIncorrectasException {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(3, 6));
		arbitro.mover(fabricarJugada(tablero, 3, 6, 6, 6));
		comprobarReyAlcanzaProvincia();
	}
}
