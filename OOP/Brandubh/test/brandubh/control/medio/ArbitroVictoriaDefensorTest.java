package brandubh.control.medio;

import static brandubh.control.TestUtil.fabricarJugada;
import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertAll;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import brandubh.control.Arbitro;
import brandubh.modelo.Pieza;
import brandubh.modelo.Tablero;
import brandubh.util.Color;
import brandubh.util.Coordenada;
import brandubh.util.TipoPieza;

/**
 * Comprobación de victoria del defensor alcanzando provincias.
 * 
 * @author <a href="rmartico@ubu.es">Raúl Marticorena</a>
 * @since 1.0
 * @version 1.0
 */
@DisplayName("Tests del Arbitro sobre victorias del defensor alcanzando provincias.")
@Timeout(value = 2, unit = TimeUnit.SECONDS) // Time out global para todos los tests salvo los de ciclo de vida
public class ArbitroVictoriaDefensorTest {

	/** Árbitro de testing. */
	private Arbitro arbitro;

	/** Tablero de testing. */
	private Tablero tablero;

	/** Generación del árbitro para testing. */
	@BeforeEach
	void inicializar() {
		// Inyección de tablero para testing...
		tablero = new Tablero();
		arbitro = new Arbitro(tablero);
		arbitro.cambiarTurno(); // dejamos el turno en defensor
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
	 */
	@Test
	@DisplayName("Comprueba que se alcanza la esquina superior izquierda en horizontal por el rey.")
	void comprobarAlcanceEsquinaSuperiorIzquierdaEnHorizontal() {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(0, 3));
		arbitro.mover(fabricarJugada(tablero, 0, 3, 0, 0));
		comprobarReyAlcanzaProvincia();
	}
	
	/**
	 * Comprueba que se alcanza la esquina superior derecha en horizontal por el rey.
	 */
	@Test
	@DisplayName("Comprueba que se alcanza la esquina superior derecha en horizontal por el rey.")
	void comprobarAlcanceEsquinaSuperiorDerechaEnHorizontal() {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(0, 3));
		arbitro.mover(fabricarJugada(tablero, 0, 3, 0 , 6));
		comprobarReyAlcanzaProvincia();
	}
	
	/**
	 * Comprueba que se alcanza la esquina superior izquierda en horizontal por el rey.
	 */
	@Test
	@DisplayName("Comprueba que se alcanza la esquina inferior izquierda en horizontal por el rey.")
	void comprobarAlcanceEsquinaInferiorIzquierdaEnHorizontal() {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(6, 3));
		arbitro.mover(fabricarJugada(tablero, 6, 3, 6, 0));
		comprobarReyAlcanzaProvincia();
	}
	
	/**
	 * Comprueba que se alcanza la esquina inferior derecha en horizontal por el rey.
	 */
	@Test
	@DisplayName("Comprueba que se alcanza la esquina inferior derecha en horizontal por el rey.")
	void comprobarAlcanceEsquinaInferiorDerechaEnHorizontal() {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(6, 3));
		arbitro.mover(fabricarJugada(tablero, 6, 3, 6, 6));
		comprobarReyAlcanzaProvincia();
	}
	
	/**
	 * Comprueba que se alcanza la esquina superior izquierda en vertical por el rey.
	 */
	@Test
	@DisplayName("Comprueba que se alcanza la esquina superior izquierda en vertical por el rey.")
	void comprobarAlcanceEsquinaSuperiorIzquierdaEnVertical() {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(3, 0));
		arbitro.mover(fabricarJugada(tablero, 3, 0, 0, 0));
		comprobarReyAlcanzaProvincia();
	}
	
	/**
	 * Comprueba que se alcanza la esquina superior derecha en vertical por el rey.
	 */
	@Test
	@DisplayName("Comprueba que se alcanza la esquina superior derecha en vertical por el rey.")
	void comprobarAlcanceEsquinaSuperiorDerechaEnVertical() {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(3, 6));
		arbitro.mover(fabricarJugada(tablero, 3, 6, 0, 6));
		comprobarReyAlcanzaProvincia();
	}
	
	/**
	 * Comprueba que se alcanza la esquina inferior izquierda en vertical por el rey.
	 */
	@Test
	@DisplayName("Comprueba que se alcanza la esquina inferior izquierda en vertical por el rey.")
	void comprobarAlcanceEsquinaInferiorIzquierdaEnVertical() {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(3, 0));
		arbitro.mover(fabricarJugada(tablero, 3, 0, 6, 0));
		comprobarReyAlcanzaProvincia();
	}
	
	/**
	 * Comprueba que se alcanza la esquina inferior derecha en vertical por el rey.
	 */
	@Test
	@DisplayName("Comprueba que se alcanza la esquina superior derecha en vertical por el rey.")
	void comprobarAlcanceEsquinaInferiorDerechaEnVertical() {		
		tablero.colocar(new Pieza(TipoPieza.REY), new Coordenada(3, 6));
		arbitro.mover(fabricarJugada(tablero, 3, 6, 6, 6));
		comprobarReyAlcanzaProvincia();
	}
}
