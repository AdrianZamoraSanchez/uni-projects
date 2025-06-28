package juego.modelo;
import juego.util.*;

public class Tablero{
	public Celda matriz[][];
	
	public Tablero(int filas, int columnas) {
		this.matriz = new Celda[filas][columnas];
		
		// Inicializa cada celda en la matriz
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                matriz[i][j] = new Celda(new Coordenada(i,j));
            }
        }
	}
	
	public void aTexto(){
		for(int i = 0; i < this.matriz.length; i++) {
			for(int j = 0; j < this.matriz[0].length; j++) {	
				System.out.printf("%s \t", this.matriz[i][j].toString());
			}
			System.out.printf("\n");
		}
	}
	
	public boolean estaCompleto(){
		for(int i = 0; i < this.matriz.length; i++) {
			for(int j = 0; j < this.matriz[0].length; j++) {
				if(this.matriz[i][j].estaVacia()) {
					return false;
				}
			}
		}
		return true;
	}
	
	public void colorcar(Pieza pieza, Coordenada coordenada){
		this.matriz[coordenada.fila()][coordenada.columna()].establecerPieza(pieza);
	}
	
	public int consultarNumeroColumnas() {
		return matriz[0].length;
	}
	public int consultarNumeroFilas() {
		return matriz.length;
	}
}