package tafl.modelo;

import java.util.Objects;

import tafl.util.*;

/**
 * Clase que implementa las piezas del juego, estas serán colocadas
 * en el tablero, estarán indentificadas por su tipo
 * 
 * @author <a href="azs1004@alu.ubu.es">Adrián Zamora Sánchez</a>
 * @see tafl.control.Arbitro
 * @see tafl.util.TipoPieza
 * @version 1.0
 * @since 1.0
*/

public class Pieza {
	/**
	 * Tipo de pieza correspondiente a esta pieza
	 * 
	 * @see tafl.util.TipoPieza
	 */
	private TipoPieza tipoPieza;
	
	/**
	 * Contructor de pieza, debe recibir un tipo de pieza
	 * 
	 * @param tipo tipo de pieza
	 */
	public Pieza(TipoPieza tipo) {
		tipoPieza = tipo;
	}
	
	/**
	 * Clona la pieza actual y devuelve una exactamente igual
	 * 
	 * @return Pieza pieza identica a la actual
	 */
	public Pieza clonar() {
		// Genera una nueva pieza con el mismo tipo que la actual
		Pieza pieza = new Pieza(tipoPieza);
		return pieza;
	}
	
	/**
	 * Devuelve el tipo de pieza
	 * 
	 * @return TipoPieza
	 */
	public TipoPieza consultarTipoPieza(){
		return tipoPieza;
	}
	
	/**
	 * Devuelve el color de la pieza
	 * 
	 * @return Color
	 */
	public Color consultarColor() {
		return tipoPieza.consultarColor();
	}
	
	/**
	 * Devuelve un string con los datos de la pieza
	 * 
	 * @return String
	 */
	public String toString() {
		String datos = "Pieza [tipoPieza=[" + tipoPieza.toString() + "]]" ;
		return datos;
	}

	/**
	 * Devuelve el codigo hash que identifica unequivocamente cada instancia de este objeto
	 * 
	 * @return int codigo hash
	 */
	@Override
	public int hashCode() {
		return Objects.hash(tipoPieza);
	}

	/**
	 * Devuelve true si las piezas son iguales y false si son distintas
	 * 
	 * @param obj objeto con el que se compara
	 * @return boolean
	 */
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Pieza other = (Pieza) obj;
		return tipoPieza == other.tipoPieza;
	}
}