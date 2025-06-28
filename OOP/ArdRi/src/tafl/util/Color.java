package tafl.util;

/**
 * Clase que implementa el color de las piezas
 * 
 * @author <a href="azs1004@alu.ubu.es">Adrián Zamora Sánchez</a>
 * @see tafl.modelo.Pieza
 * @version 1.0
 * @since 1.0
 * 
*/
public enum Color {
	
	/**
	 * Color de piezas blancas, piezas defensoras (D) y rey (R)
	 */
	BLANCO('B'),
	/**
	 * Color de piezas negras, piezas atacantes (A)
	 */
	NEGRO('N');
	
	/**
	 * Letra correspondiente al color
	 * 
	 * @see java.lang.Object
	 */
	private char letra;
	
	/**
	 * Constructor de color, asigna a letra su caracter correspondiente
	 * 
	 * @param letra caracter correspondiente a cada color
	 */
	private Color(char letra) {
		this.letra = letra;
	}
	
	/**
	 * Método que devuelve la letra correspondiente al color
	 * 
	 * @return char
	 */
	public char toChar() {
		return letra;
	}
	
	/**
	 * Método que devuelve el color de contrario al actual
	 * 
	 * @return Color
	 */
	public Color consultarContrario() {
		Color color;
		
		// Si la letra del color actual es blanco (caracter 'B') debe devolver el color negro,
		if(this.letra == 'B') {
			color = Color.NEGRO;
			return color;
		}
		
		// En caso contrario devuelve el color blanco
		color = Color.BLANCO;
		return color;
		
	}
}
	
	
