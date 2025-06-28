package brandubh.util;

/**
 * Clase que implementa el traductor entre notación algebraica y posicional
 * basada en las coordenadas del tablero de Brandubh
 * 
 * @author <a href="azs1004@alu.ubu.es">Adrián Zamora Sánchez</a>
 * @version 1.0
 * @since 1.0
*/
public class Traductor{
	
	/**
	 * Constructor de traductor (no hace nada)
	 */
	public Traductor() {}
	
	/**
	 * Toma una cadena de texto en notación algebraica y la traduce a notación 
	 * de coordenadas (x,y)
	 * 
	 * @param texto texto con el formato en notación algebraica
	 * @return coordenada coordenadas correspondientes a la notación algebraica
	*/
	public static Coordenada consultarCoordenadaParaNotacionAlgebraica(String texto) {
		// Se comprueba si es correcto el texto
		if(!esTextoCorrectoParaCoordenada(texto)) {
			return null;
		}
		
		// Se toman la letra y el numero
		char letra = texto.charAt(0);
        char numeroChar = texto.charAt(1);
		
        // Se generan las coordenadas
		int fila = 7 - (numeroChar - '0');
        int columna = letra - 'a';
		
        // Return de la coordenada en formato coordenadas cartesianas
		return new Coordenada(fila, columna);
	}
	
	/**
	 * Toma un coordenada (x,y) y la pasa a notación algebraica [0-7][a-g]
	 * 
	 * @param coordenada coordenada a traducir
	 * @return String texto con las coordenadas traducidas a notación algebraica
	*/
	public static String consultarTextoEnNotacionAlgebraica(Coordenada coordenada) {
	    // Se comprueba si es correcta la coordenada
		if (coordenada.fila() > 6 || coordenada.columna() > 6 || coordenada.fila() < 0 || coordenada.columna() < 0) {
	        return null;
	    }
	    
		// Se define la variable que almacena las coordenadas en notación algebraica
	    String coordenadasEnTexto = "";
	    
	    // Se escribe en notación algebraica las coordenadas
	    char letra = (char) ('a' + coordenada.columna());
	    coordenadasEnTexto += letra;
	    coordenadasEnTexto += (char) ('0' + 7 - coordenada.fila());
	    
	    // Return de la coordenada en notación algebraica
	    return coordenadasEnTexto;
	}

	/**
	 * Comprueba si un texto de posición en notación algebraica es correcto como coordenadas
	 * 
	 * @param texto coordenada a comprobar
	 * @return boolean true is es correcto, false si no lo es
	*/
	public static boolean esTextoCorrectoParaCoordenada(String texto) {
		// Debe tener una longitud NO mayor que dos
		if(texto.length() > 2) {
			return false;
		}
		
		// Si es una letra menor que 'a' o mayor que 'g' en la tabla ASCII es incorrecto
		if(texto.charAt(0) < 'a' || texto.charAt(0) > 'g') {
			return false;
		}
		
		// Si el numero es menor que 1 o mayor que 7 es una coordenada incorrecta
		if(texto.charAt(1) < '1' || texto.charAt(1) > '7') {
			return false;
		}
		
		// Si no hay fallos en el formato la coordenada es correcta
		return true;
	}
}