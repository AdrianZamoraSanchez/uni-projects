package tafl.modelo;

import java.util.Objects;

import tafl.util.*;

/**
 * Clase que implementa las celdas que conforman el tablero, estas tendran
 * distintos tipos y contendrán las piezas del juego.
 * 
 * @author <a href="azs1004@alu.ubu.es">Adrián Zamora Sánchez</a>
 * @see tafl.modelo.Tablero
 * @see tafl.modelo.Pieza
 * @see tafl.util.TipoCelda
 * @version 1.0
 * @since 1.0
 * 
*/
public class Celda {
	/**
	 * Pieza contenida en la celda
	 * 
	 * @see tafl.modelo.Pieza
	 */
	private Pieza pieza;
	
	/**
	 * Tipo de celda, por defecto es NORMAL
	 * 
	 * @see tafl.util.TipoCelda
	 */
	private TipoCelda tipoCelda = TipoCelda.NORMAL;
	
	/**
	 * Coordenadas de la celda en el tablero
	 * 
	 * @see tafl.util.Coordenada
	 */
	private Coordenada coordenada;
	
	/**
	 * Constructor de Celda, asigna las coordenadas a la celda y su tipo
	 * 
	 * @param coordenadaParam coordenada correspondiente a la celda
	 * @param tipoCeldaParam el tipo de celda que le corresponde
	 */
	public Celda(Coordenada coordenadaParam, TipoCelda tipoCeldaParam) {
		coordenada = coordenadaParam;
		tipoCelda = tipoCeldaParam;
	}
	
	/**
	 * Constructor de Celda, asigna las coordenadas a la celda,
	 * se utiliza para las celdas normales, con el tipoCelda ya definido
	 * 
	 * @param coordenadaParam coordenada correspondiente a la celda
	 */
	public Celda(Coordenada coordenadaParam) {
		coordenada = coordenadaParam;
	}
	
	/**
	 * Devuelve las coordenadas de la celda
	 * 
	 * @return coordenada
	 */
	public Coordenada consultarCoordenada() {
		return coordenada;
	}
	
	/**
	 * Devuelve el color de la pieza que contiene
	 * 
	 * @return Color
	 */
	public Color consultarColorDePieza() {
		if(pieza == null) {
			return null;
		}
		return pieza.consultarColor();
	}
	
	/**
	 * Devuelve la pieza que contiene la celda
	 * 
	 * @return Pieza
	 */
	public Pieza consultarPieza() {
		return pieza;
	}
	
	/**
	 * Coloca una pieza en la celda
	 * 
	 * @param piezaAColocar valor de la pieza que se coloca
	 */
	public void colocar(Pieza piezaAColocar) {
		pieza = piezaAColocar;
	}
	
	/**
	 * Devuelve el tipo de celda de esta celda
	 * 
	 * @return TipoCelda
	 */
	public TipoCelda consultarTipoCelda() {
		return tipoCelda;
	}
	
	/**
	 * Elimina la pieza que contenga esta celda
	 */
	public void eliminarPieza() {
		// Da el valor null a la pieza
		pieza = null;
	}
	
	/**
	 * Clona la celda y sus atributos y devuelve una celda
	 * exactamente igual
	 * 
	 * @return Celda
	 */
	public Celda clonar() {
		// Genera una nueva celda con las coordenadas y tipo de la celda actual
        Celda clon = new Celda(coordenada, tipoCelda);
        
        // Se comprueba que la celda actual contenga alguna pieza
        if (pieza != null) {
        	// Si tiene una pieza la clona y añade al objeto clon
        	clon.pieza = pieza.clonar();
        }
        
        // Devuelve el objeto clonado
        return clon;
    }
	
	/**
	 * Devuelve true si está vacía y false si contiene una pieza
	 * 
	 * @return boolean
	 */
	public boolean estaVacia() {
		// Si no hay pieza devuelve true
		if(pieza == null) {
			return true;
		}
		
		// Si hay pieza devuelve false
		return false;
	}
	
	/**
	 * Devuelve el codigo hash que identifica unequivocamente cada instancia de este objeto
	 * 
	 * @return int codigo hash
	 */
	@Override
	public int hashCode() {
		return Objects.hash(coordenada, pieza, tipoCelda);
	}

	/**
	 * Devuelve true si el objeto pasado como párametro es igual al actual
	 * 
	 * @param obj objeto a comparar
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
		Celda other = (Celda) obj;
		return Objects.equals(coordenada, other.coordenada) && Objects.equals(pieza, other.pieza)
				&& tipoCelda == other.tipoCelda;
	}
	
	/**
	 * Devuelve los datos de esta celda
	 * 
	 * @return String
	 */
	public String toString() {
	    if (estaVacia()) {
	    	 return "Celda [pieza=null, coordenada=" + coordenada.toString() + "]";
	    } else {
	        return "Celda [pieza=" + consultarPieza().toString() + ", coordenada=" + coordenada.toString() + "]";
	    }
	}
}