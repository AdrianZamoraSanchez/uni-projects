package ubu.adrian.taller.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;

/**
 * Entidad que modela la relación entre eventos y sus 
 * categorias asociadas
 */
@Entity
@Table(name = "events_categories")
public class EventCategory {

	// Id de la asociación evento-categoría
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private long id;

    // Categoría asociada al evento
    @Enumerated(EnumType.STRING)
    @Column(name = "category", nullable = false, length = 50)
    private Categories category;

    // Evento que se asocia con la categoría
    @ManyToOne
    @JoinColumn(name = "event_id", nullable = false)
    private Event event;

    /**
     * Constructor con categoría y evento
     * 
     * @param category Categoria de la relacion
     * @param event    Evento cuya categoria se guarda
     */
    public EventCategory(Categories category, Event event) {
        this.category = category;
        this.event = event;
    }
    
    /**
     * Constructor por defecto
     */
    public EventCategory() {}
    
    /**
     * Establece la categoria
     * 
     * @param category Categoría que se asigna a la relación
     */
    public void setCategory(Categories category) {
    	this.category = category;
    }
    
    /**
     * Devuelve la categoria
     * 
     * @return category categoria de la relacion
     */
    public Categories getCategory() {
    	return category;
    }
    
    /**
     * Establece el evento asociado 
     * 
     * @param event Evento que forma parte de la relación
     */
    public void setEvent(Event event) {
    	this.event = event;
    }
    
    /**
     * Devuelve el evento asociado 
     * 
     * @return event Evento asociado a esta categoria
     */
    public Event getEvent() {
    	return event;
    }
}
