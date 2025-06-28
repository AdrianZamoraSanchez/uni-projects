package brandubh.util;

/**
 * Enumeración que implementa los tipos de piezas rey, atacante y defensor
 * 
 * @author <a href="azs1004@alu.ubu.es">Adrián Zamora Sánchez</a>
 * @see brandubh.modelo.Pieza
 * @version 1.0
 * @since 1.0
 * 
*/
public enum TipoPieza {
	
	/**
	 * Tipo de pieza de los defensores, le corresponde el color blanco
	 */
	DEFENSOR('D'),
	/**
	 * Tipo de pieza de los atacantes, le corresponde el color negro
	 */
	ATACANTE('A'),
	
	/**
	 * Tipo de pieza del rey, le corresponde el color blanco
	 */
	REY('R');
	
	/**
	 * Caracter que identifica los tipos de piezas
	 * 
	 * @see brandubh.util.Color
	 */
	private Color color;
	
	/**
	 * Caracter que identifica los tipos de piezas
	 * 
	 * @see java.lang.Object
	 */
	private char caracter;
	
	/**
	 * Constructor de TipoPieza, asocia cada tipo con el caracter correspondiente
	 * 
	 * @param caracterEntrada caracter correspondiente a cada color
	 */
	private TipoPieza(char caracterEntrada) {
		caracter = caracterEntrada;
		
		// Si no se pasa parámetro de color se establece comprobando el caracter de la pieza
		if(caracter == 'A') {
			// Si no es atacante es pieza negra
			color = Color.NEGRO;
		}else {
			// Si no es atacante (rey o defensor) será pìeza blanca
			color = Color.BLANCO;
		}
	}
	
	/**
	 * Constructor de TipoPieza, asocia cada tipo con el caracter correspondiente
	 * 
	 * @param caracterEntrada caracter correspondiente a cada color
	 */
	private TipoPieza(char caracterEntrada, Color colorEntrada) {
		caracter = caracterEntrada;
		color = colorEntrada;
	}
	
	/**
	 * Devuelve el color de la pieza
	 * 
	 * @return Color
	 */
	public Color consultarColor() {
		return color;
	}
	
	/**
	 * Devuelve el caracter correspondiente al tipo de pieza
	 * 
	 * @return char
	 */
	public char toChar() {
		return caracter;
	}
}
	
	
