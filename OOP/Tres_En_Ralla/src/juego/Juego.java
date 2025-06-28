package juego;
import juego.modelo.*;
import juego.util.*;

public class Juego{
	public static void main(String[] args) {
		Tablero tablero = new Tablero(3,4); // Filas, Columnas
		
		Pieza pieza1 = new Pieza(Color.NEGRO);
		Pieza pieza2 = new Pieza(Color.BLANCO);
		Pieza pieza3 = new Pieza(Color.NEGRO);
		
		Coordenada coordenada1 = new Coordenada(0,0);
		Coordenada coordenada2 = new Coordenada(1,1);
		Coordenada coordenada3 = new Coordenada(2,2);
		
		tablero.colorcar(pieza1, coordenada1);
		tablero.colorcar(pieza2, coordenada2);
		tablero.colorcar(pieza3, coordenada3);
		
		System.out.printf("Numero filas: %d\nNumero columnas %d\n", tablero.consultarNumeroFilas(),tablero.consultarNumeroColumnas());
		
		tablero.aTexto();
	}
}