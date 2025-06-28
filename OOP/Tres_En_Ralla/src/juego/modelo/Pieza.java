package juego.modelo;

public class Pieza{
	Color color;
	
	public Pieza(Color color) {
		this.color = color;
	}
	public String aTexto(){
		return this.color.toString();
	}
	public Color obtenerColor() {
		return this.color;
	}
	
	public String toString() {
		String datos = "Pieza [color = Color[" + this.color + "]]" ;
		return datos;
	}
}