package tafl.excepcion;

/**
 * Excepción que indica que una coordenada no es correcta
 * 
 * @author <a href="azs1004@alu.ubu.es">Adrián Zamora Sánchez</a>
 * @see tafl.util.Coordenada
 * @version 1.0
 * @since 1.0
*/
public class CoordenadasIncorrectasException extends Exception{
	
	/**
	 * Serial de la excepción
	 * 
	 * @see java.lang.Object
	 */
	private static final long serialVersionUID = 1L;
	
	/**
	 * Constructor por defecto
	 */
	public CoordenadasIncorrectasException() {
        super("Excepción de coordenadas incorrectas");
    }

	/**
	 * Constructor
	 * 
	 * @param message mensaje del error
	 */
    public CoordenadasIncorrectasException(String message) {
        super(message);
    }

    /**
	 * Constructor
	 * 
	 * @param cause excepción que se ha lanzado
	 */
    public CoordenadasIncorrectasException(Throwable cause) {
        super("Excepción de coordenadas incorrectas", cause);
    }

    /**
   	 * Constructor
   	 * 
   	 * @param message mensaje del error
   	 * @param cause excepción que se ha lanzado
   	 */
    public CoordenadasIncorrectasException(String message, Throwable cause) {
        super(message, cause);
    }
}