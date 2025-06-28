package tafl.excepcion;

/**
 * Excepción que indica que un arbitro no es correcto
 * 
 * @author <a href="azs1004@alu.ubu.es">Adrián Zamora Sánchez</a>
 * @see tafl.control.Arbitro
 * @version 1.0
 * @since 1.0
*/
public class TipoArbitroException extends Exception{
	/**
	 * Serial de la excepción
	 * 
	 * @see java.lang.Object
	 */
	private static final long serialVersionUID = 1L;
	
	/**
	 * Constructor por defecto
	 */
	public TipoArbitroException() {
		super("Excepción de arbitro incorrecto");
	}
	
	/**
   	 * Constructor
   	 * 
   	 * @param message mensaje del error
   	 */
	public TipoArbitroException(String message) {
		super(message);
	}
	
	/**
   	 * Constructor
   	 * 
   	 * @param cause excepción que se ha lanzado
   	 */
	public TipoArbitroException(Throwable cause) {
		super("Excepción de arbitro incorrecto", cause);
	}
	
	/**
   	 * Constructor
   	 * 
   	 * @param message mensaje del error
   	 * @param cause excepción que se ha lanzado
   	 */
	public TipoArbitroException(String message, Throwable cause) {
		super(message, cause);
	}
}