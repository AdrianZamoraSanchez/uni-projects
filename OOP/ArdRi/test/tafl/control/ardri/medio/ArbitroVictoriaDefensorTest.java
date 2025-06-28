package tafl.control.ardri.medio;

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

import tafl.control.ArbitroArdRi;
import tafl.excepcion.CoordenadasIncorrectasException;
import tafl.modelo.Pieza;
import tafl.modelo.Tablero;
import tafl.util.Color;
import tafl.util.Coordenada;
import tafl.util.TipoPieza;

/**
 * Comprobación del ArbitroArdRi de victoria del defensor alcanzando provincias.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0.0.1
 */
@DisplayName("Tests del ArbitroArdRi sobre victorias del defensor alcanzando provincias.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public class ArbitroVictoriaDefensorTest extends tafl.control.medio.ArbitroAbstractoVictoriaDefensorTest {

	/** Generación del árbitro para testing. */
	@BeforeEach
	void inicializar() {
		// Inyección de tablero para testing...
		tablero = new Tablero();
		arbitro = new ArbitroArdRi(tablero);
		arbitro.cambiarTurno(); // dejamos el turno en defensor
		// y NO colocamos piezas para mover directamente el rey sobre el tablero...
	}
	
	/**
	 * Comprueba condiciones de victoria al alcanzar el rey el borde.
	 */
	private void comprobarReyAlcanzaBorde() {
		assertAll("victoria alcanzando borde",
				() -> assertThat("No debería ganar el atacante.", arbitro.haGanadoAtacante(), is(false)),
				() -> assertThat("Debería ganar el defensor.", arbitro.haGanadoRey(), is(true)),
				() -> assertThat("El turno debería ser del jugado defensor.", arbitro.consultarTurno(),
						is(Color.BLANCO)));
	}
	
	/**
	 * Comprueba que se alcanza borde superior por el rey.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 */
	@Test
	@DisplayName("Comprobar que se alcanza el borde superior por el rey.")
	void comprobarAlcanceBordeSuperior()  throws CoordenadasIncorrectasException {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(1, 3));
		arbitro.mover(fabricarJugada(tablero, 1, 3, 0, 3));
		comprobarReyAlcanzaBorde();
	}
	
	/**
	 * Comprueba que se alcanza borde inferior por el rey.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 */
	@Test
	@DisplayName("Comprobar que se alcanza el borde inferior por el rey.")
	void comprobarAlcanceBordeInferior()  throws CoordenadasIncorrectasException {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(5, 3));
		arbitro.mover(fabricarJugada(tablero, 5, 3, 6, 3));
		comprobarReyAlcanzaBorde();
	}
	
	/**
	 * Comprueba que se alcanza borde izquierdo por el rey.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 */
	@Test
	@DisplayName("Comprobar que se alcanza el borde izquierdo por el rey.")
	void comprobarAlcanceBordeIzquierdo()  throws CoordenadasIncorrectasException {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(3, 1));
		arbitro.mover(fabricarJugada(tablero, 3, 1, 3, 0));
		comprobarReyAlcanzaBorde();
	}
	
	/**
	 * Comprueba que se alcanza borde derecho por el rey.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 */
	@Test
	@DisplayName("Comprobar que se alcanza el borde derecho por el rey.")
	void comprobarAlcanceBordeDerecho()  throws CoordenadasIncorrectasException {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(3, 5));
		arbitro.mover(fabricarJugada(tablero, 3, 5, 3, 6));
		comprobarReyAlcanzaBorde();
	}	
}
