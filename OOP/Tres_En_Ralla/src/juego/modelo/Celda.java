package juego.modelo;
import juego.util.*;

public class Celda{
	Pieza pieza;
	Coordenada coordenadas;
	
	public Celda(Coordenada coordenada) {
		this.coordenadas = coordenada;
	}
	
	public Coordenada consultarCoordenada() {
		return this.coordenadas;
	}
	
	public boolean estaVacia() {
		if(pieza == null) {
			return true;
		}else {
			return false;
		}
		
	}
	
	public void establecerPieza(Pieza pieza) {
		this.pieza = pieza;
	}
	
	
	public Pieza obtenerPieza() {
		return this.pieza;
	}
	
	public String toString() {
	    if (estaVacia()) {
	        return "[ ]";
	    } else {
	        return "[" + obtenerPieza().obtenerColor().toChar() + "]";
	    }
	}
}