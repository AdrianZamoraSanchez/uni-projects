package tafl.control.brandubh.basico;

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
import tafl.control.basico.ArbitroAbstractoInicializacionTest;
import tafl.excepcion.CoordenadasIncorrectasException;
import tafl.modelo.Jugada;
import tafl.modelo.Tablero;

/**
 * Comprobación del ArbitroBrandubh movimientos ilegales.
 * 
 * Se asume que al menos se han colocado correctamente las piezas. En caso
 * contrario estos tests no se pasarán.
 * 
 * @see ArbitroAbstractoInicializacionTest
 */
@DisplayName("Tests del ArbitroBrandubh sobre el control de movimientos ilegales.")
@Timeout(value = 2, unit = TimeUnit.SECONDS, threadMode = SEPARATE_THREAD) // Time out global para todos los tests salvo los de ciclo de vida
public class ArbitroMovimientosIlegalesTest extends tafl.control.basico.ArbitroAbstractoMovimientosIlegalesTest {
	
	/** Generación del árbitro para testing. */
	// @formatter:off
	/* Partimos de un tablero con las piezas ya colocadas como el que se muestra:	
	 * 7 - - - A - - -  
	 * 6 - - - A - - - 
	 * 5 - - - D - - - 
	 * 4 A A D R D A A 
	 * 3 - - - D - - -
	 * 2 - - - A - - - 
	 * 1 - - - A - - -
	 *   a b c d e f g   
	 */
	 // @formatter:on
	@BeforeEach
	void inicializar() {
		// Inyección de tablero para testing...
		tablero = new Tablero();
		arbitro = new ArbitroBrandubh(tablero);
		arbitro.colocarPiezasConfiguracionInicial();
	}
	
	/**
	 * Comprobacion de ilegalidad de intentar mover una pieza atacante a una
	 * provincia al inicio de partida. 
	 * 
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento ilegal de pieza atacante a una provincia iniciando partida.")
	@Test
	void comprobarPiezaAtacanteAUnaProvinciaIniciandoPartida() {
		// given
		// then
		Jugada jugada1 = fabricarJugada(tablero, 0, 3, 0, 6);
		Jugada jugada2 = fabricarJugada(tablero, 3, 6, 6, 6);
		Jugada jugada3 = fabricarJugada(tablero, 6, 3, 6, 0);
		Jugada jugada4 = fabricarJugada(tablero, 3, 0, 0, 0);
		Jugada jugada5 = fabricarJugada(tablero, 0, 3, 0, 0);
		Jugada jugada6 = fabricarJugada(tablero, 3, 0, 6, 0);
		Jugada jugada7 = fabricarJugada(tablero, 6, 3, 6, 6);
		Jugada jugada8 = fabricarJugada(tablero, 3, 6, 0, 6);
		assertAll("mover atacante sobre provincia",
				() -> assertThat("El movimiento debería ser ilegal para " + jugada1, arbitro.esMovimientoLegal(jugada1),
						is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada2, arbitro.esMovimientoLegal(jugada2),
						is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada3, arbitro.esMovimientoLegal(jugada3),
						is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada4, arbitro.esMovimientoLegal(jugada4),
						is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada5, arbitro.esMovimientoLegal(jugada5),
						is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada6, arbitro.esMovimientoLegal(jugada6),
						is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada7, arbitro.esMovimientoLegal(jugada7),
						is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada8, arbitro.esMovimientoLegal(jugada8),
						is(false)));
	}


	/**
	 * Comprobacion de ilegalidad de intentar mover defensor en el
	 * turno inicial que no le corresponde.
	 * 
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento ilegal de mover defensor en el turno inicial.")
	@Test
	void comprobarPiezaAtacanteSobrePiezaDefensoraContigua() {
		Jugada jugada1 = fabricarJugada(tablero,2,3,2,4);
		Jugada jugada2 = fabricarJugada(tablero,3,2,2,2);
		Jugada jugada3 = fabricarJugada(tablero,4,3,4,5);
		Jugada jugada4 = fabricarJugada(tablero,3,4,1,4);
		assertAll("mover defensor en el turno que no le corresponde",
				() -> assertThat("El movimiento debería ser ilegal para " + jugada1, arbitro.esMovimientoLegal(jugada1), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada2, arbitro.esMovimientoLegal(jugada2), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada3, arbitro.esMovimientoLegal(jugada3), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada4, arbitro.esMovimientoLegal(jugada4), is(false))			
				);
	}
	
	/**
	 * Comprobacion de ilegalidad de intentar mover una pieza defensora
	 * sobre otra pieza atacante, avanzando el turno actual
	 * con piezas defensoras.
	 * 
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento ilegal de pieza defensora sobre pieza atacante contigua.")
	@Test
	void comprobarPiezaDefensoraSobrePiezaAtacanteContigua() {
		// given
		// when
		arbitro.cambiarTurno(); 
		// then
		Jugada jugada1 = fabricarJugada(tablero,3,2,3,1);
		Jugada jugada2 = fabricarJugada(tablero,2,3,1,3);
		Jugada jugada3 = fabricarJugada(tablero,3,4,3,5);
		Jugada jugada4 = fabricarJugada(tablero,4,3,5,3);
		assertAll("mover defensor sobre pieza atacante contigua",
				() -> assertThat("El movimiento debería ser ilegal para " + jugada1, arbitro.esMovimientoLegal(jugada1), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada2, arbitro.esMovimientoLegal(jugada2), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada3, arbitro.esMovimientoLegal(jugada3), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada4, arbitro.esMovimientoLegal(jugada4), is(false))			
				);
	}
	

	/**
	 * Comprobacion de ilegalidad saltando atacante sobre defensor.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento ilegal de pieza atacante saltando sobre pieza defensora.")
	@Test
	void comprobarSaltoDeAtacanteSobreDefensor() throws CoordenadasIncorrectasException {
		// given
		// when
		arbitro.mover(fabricarJugada(tablero,0,3,0,4)); // movemos una pieza atacante 
		arbitro.cambiarTurno(); // pasamos turno a blancas
		arbitro.mover(fabricarJugada(tablero,3,4,2,4)); // movemos una pieza defensora 
		arbitro.cambiarTurno(); // retornamos turno a negro
		// then
		Jugada jugada1 = fabricarJugada(tablero,0,4,4,4); // saltar atacante sobre defensor
		assertAll("intentar saltar atacante sobre pieza defensora",
				() -> assertThat("El movimiento debería ser ilegal para " + jugada1, arbitro.esMovimientoLegal(jugada1), is(false))		
				);
	}
	
	/**
	 * Comprobacion de ilegalidad saltando defensor sobre atacante.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento ilegal de pieza defensora saltando sobre pieza atacante.")
	@Test
	void comprobarSaltoDeDefensorSobreAtacante() throws CoordenadasIncorrectasException {
		// given
		// when
		arbitro.mover(fabricarJugada(tablero,3,5,2,5)); // movemos una pieza atacante
		arbitro.cambiarTurno(); // turno para el defensor blanco
		// then
		Jugada jugada1 = fabricarJugada(tablero, 2,3,2,6); // saltar defensor sobre atacante
		assertAll("intentar saltar defensor sobre pieza atacante",
				() -> assertThat("El movimiento debería ser ilegal para " + jugada1, arbitro.esMovimientoLegal(jugada1), is(false))		
				);
	}
	
	
	
	/**
	 * Comprobacion de ilegalidad de intentar mover en diagonal una pieza
	 * atacante.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento ilegal de pieza atacante en diagonal.")
	@Test
	void comprobarPiezaAtacanteEnDiagonal() throws CoordenadasIncorrectasException {
		Jugada jugada1 = fabricarJugada(tablero,3,5,2,6);
		Jugada jugada2 = fabricarJugada(tablero,0,3,2,1);
		Jugada jugada3 = fabricarJugada(tablero,3,1,4,2);
		Jugada jugada4 = fabricarJugada(tablero,6,3,4,5);
		assertAll("mover atacante en diagonal",
				() -> assertThat("El movimiento debería ser ilegal para " + jugada1, arbitro.esMovimientoLegal(jugada1), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada2, arbitro.esMovimientoLegal(jugada2), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada3, arbitro.esMovimientoLegal(jugada3), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada4, arbitro.esMovimientoLegal(jugada4), is(false))			
				);
	}
	

	/**
	 * Comprobacion de ilegalidad de intentar mover en diagonal una pieza
	 * defensora.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento ilegal de pieza defensora en diagonal.")
	@Test
	void comprobarPiezaDefensoraEnDiagonal() throws CoordenadasIncorrectasException {
		// given
		// when
		arbitro.cambiarTurno(); // pasamos turno a defensor con blancas
		// then
		Jugada jugada1 = fabricarJugada(tablero,2,3,0,5);
		Jugada jugada2 = fabricarJugada(tablero,3,4,4,5);
		Jugada jugada3 = fabricarJugada(tablero,4,3,6,1);
		Jugada jugada4 = fabricarJugada(tablero,3,2,2,1);
		assertAll("mover defensor en diagonal",
				() -> assertThat("El movimiento debería ser ilegal para " + jugada1, arbitro.esMovimientoLegal(jugada1), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada2, arbitro.esMovimientoLegal(jugada2), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada3, arbitro.esMovimientoLegal(jugada3), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada4, arbitro.esMovimientoLegal(jugada4), is(false))			
				);
	}
	
	/**
	 * Comprobacion de ilegalidad de intentar mover en diagonal al rey.
	 * 
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento ilegal del rey en diagonal.")
	@Test
	void comprobarReyEnDiagonal() {
		// given
		// when
		arbitro.cambiarTurno(); // cambiamos turno a defensores con blancas
		// then
		Jugada jugada1 = fabricarJugada(tablero,3,3,1,1);
		Jugada jugada2 = fabricarJugada(tablero,3,3,1,5);
		Jugada jugada3 = fabricarJugada(tablero,3,3,5,1);
		Jugada jugada4 = fabricarJugada(tablero,3,3,5,5);
		assertAll("mover rey en diagonal",
				() -> assertThat("El movimiento debería ser ilegal para " + jugada1, arbitro.esMovimientoLegal(jugada1), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada2, arbitro.esMovimientoLegal(jugada2), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada3, arbitro.esMovimientoLegal(jugada3), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada4, arbitro.esMovimientoLegal(jugada4), is(false))			
				);
	}
	

	/**
	 * Comprobacion de ilegalidad de intentar mover una pieza defensora
	 * a una provincia.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento ilegal de pieza defensora a una provincia.")
	@Test
	void comprobarPiezaDefensoraAUnaProvincia() throws CoordenadasIncorrectasException {
		// given
		arbitro.cambiarTurno(); // pasamos turno a defensor con blancas
		arbitro.mover(fabricarJugada(tablero,3,4,6,4)); // defensor
		// then
		Jugada jugada1 = fabricarJugada(tablero,6,4,6,6); // defensor
		assertAll("mover defensor sobre provincia",
				() -> assertThat("El movimiento debería ser ilegal para " + jugada1, arbitro.esMovimientoLegal(jugada1), is(false))
				);
	}
	
	/**
	 * Comprobacion de ilegalidad de intentar mover una pieza defensora
	 * a la celda del trono una vez abandonado.
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento ilegal de pieza defensora al trono una vez abandonado.")
	@Test
	void comprobarPiezaDefensoraAlTrono() throws CoordenadasIncorrectasException {
		// given
		// when
		arbitro.mover(fabricarJugada(tablero,3,4,4,4)); // defensor
		arbitro.mover(fabricarJugada(tablero,3,3,3,4)); // mover rey y abandonar trono		
		Jugada jugada1 = fabricarJugada(tablero,4,3,3,3); // intentar mover defensor a trono
		// then 
		arbitro.cambiarTurno(); // para que mueva el defensor
		assertThat("El movimiento debería ser ilegal para " + jugada1, arbitro.esMovimientoLegal(jugada1), is(false));
	}
	
	/**
	 * Comprobacion de ilegalidad de intentar mover el rey a la 
	 * a la celda del trono una vez abandonado porque no es su
	 * turno. 
	 * 
	 * @throws CoordenadasIncorrectasException si una coordenada está fuera del tablero
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento ilegal de rey al trono una vez abandonado porque no es su turno.")
	@Test
	void comprobarPiezaReyAlTronoUnaVezAbandonadoSinTurno() throws CoordenadasIncorrectasException {
		// given
		// when
		arbitro.mover(fabricarJugada(tablero,3,4,4,4)); // defensor
		arbitro.mover(fabricarJugada(tablero,3,3,3,4)); // mover rey y abandonar trono		
		Jugada jugada1 = fabricarJugada(tablero,3,4,3,3); // intentar mover rey a trono
		// then sin cambiar turno intentamos retonar piezas a trono
		assertThat("El movimiento debería ser ilegal para " + jugada1, arbitro.esMovimientoLegal(jugada1), is(false));
	}
	
	/**
	 * Comprobacion de ilegalidad de intentar mover una pieza defensora a su misma coordenada.
	 * 
	 * @see tafl.control.TestUtil#fabricarJugada
	 */
	@DisplayName("Comprueba el movimiento ilegal de una pieza defensora a su misma coordenada")
	@Test
	void comprobarMoverALaMismaCeldaDefensor() {
		// given
		arbitro.cambiarTurno(); // cambiamos turno
		// then
		Jugada jugada1 = fabricarJugada(tablero,2,3,2,3);
		Jugada jugada2 = fabricarJugada(tablero,3,2,3,2); 
		Jugada jugada3 = fabricarJugada(tablero,3,4,3,4); 
		Jugada jugada4 = fabricarJugada(tablero,4,3,4,3);		
		assertAll("mover defensor a su misma coordenada",
				() -> assertThat("El movimiento debería ser ilegal para " + jugada1, arbitro.esMovimientoLegal(jugada1), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada2, arbitro.esMovimientoLegal(jugada2), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada3, arbitro.esMovimientoLegal(jugada3), is(false)),
				() -> assertThat("El movimiento debería ser ilegal para " + jugada4, arbitro.esMovimientoLegal(jugada4), is(false))
				);		
	}	
}
