package ubu.adrian.taller.model;

import jakarta.persistence.*;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;

/**
 * Entidad que modela un evento
 */
@Entity
@Table(name = "events")
public class Event {
	// ID (único)
	@Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private long id;

	/**
     * Título del evento
     */
    @Column(nullable = false)
    private String title;
    
    /**
     * Descripción del evento
     */
    @Column(length = 1000)
    private String description;

    /**
     * Fecha de inicio del evento
     */
    @Column(name = "start_date", nullable = false)
    private LocalDate startDate;

    /**
     * Fecha de finalización del evento
     */
    @Column(name = "end_date", nullable = false)
    private LocalDate endDate;

    /**
     * Lugar donde se celebra el evento
     */
    @Column(nullable = false)
    private String ubication;
    
    /**
     * Imagen del evento
     */
    @Column(nullable = false)
    private String imagen;
    
    /**
     * Usuario propietario del evento
     */
    @ManyToOne
    @JoinColumn(name = "user_id", nullable = false)
    private User owner;
    
    /**
     * Número de participantes que se apuntan a este evento
     */
    @Column(name = "num_participants", nullable = false)
    private int numParticipants = 0;
    
    /**
     * Capacidad máxima de participantes (aforo del evento)
     */
    @Column(name = "max_capacity", nullable = false)
    private int maxCapacity;
    
    /**
     * Categorias del evento
     */
    @OneToMany(mappedBy = "event", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<EventCategory> categories = new ArrayList<>();
    
    /**
     * Actividades del evento
     */
    @OneToMany(mappedBy = "event", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<Activity> activities = new ArrayList<>();
    
    /**
     * Lista de participantes de este evento
     */
    @ManyToMany
    @JoinTable(
        name = "event_participants",
        joinColumns = @JoinColumn(name = "event_id"),
        inverseJoinColumns = @JoinColumn(name = "user_id")
    )
    private List<User> participants = new ArrayList<>();
    
    /**
     * Devuelve el ID del evento
     * 
     * @return ID del evento
     */
    public long getId() {
        return id;
    }

    /**
     * Establece el ID del evento
     * 
     * @param id ID del evento
     */
    public void setId(long id) {
        this.id = id;
    }

    /**
     * Devuelve el título del evento
     * 
     * @return Título del evento
     */
    public String getTitle() {
        return title;
    }

    /**
     * Establece el título del evento
     * 
     * @param title Título del evento
     */
    public void setTitle(String title) {
        this.title = title;
    }

    /**
     * Devuelve la descripción del evento
     * 
     * @return Descripción del evento
     */
    public String getDescription() {
        return description;
    }

    /**
     * Establece la descripción del evento
     * 
     * @param description Descripción del evento
     */
    public void setDescription(String description) {
        this.description = description;
    }

    /**
     * Devuelve la fecha de inicio del evento
     * 
     * @return Fecha de inicio
     */
    public LocalDate getStartDate() {
        return startDate;
    }

    /**
     * Establece la fecha de inicio del evento
     * 
     * @param startDate Fecha de inicio
     */
    public void setStartDate(LocalDate startDate) {
        this.startDate = startDate;
    }

    /**
     * Devuelve la fecha de finalización del evento
     * 
     * @return Fecha de fin
     */
    public LocalDate getEndDate() {
        return endDate;
    }

    /**
     * Establece la fecha de finalización del evento
     * 
     * @param endDate Fecha de fin
     */
    public void setEndDate(LocalDate endDate) {
        this.endDate = endDate;
    }

    /**
     * Devuelve el lugar del evento
     * 
     * @return Lugar del evento
     */
    public String getUbication() {
        return ubication;
    }

    /**
     * Establece el lugar del evento
     * 
     * @param ubication Lugar del evento
     */
    public void setUbication(String ubication) {
        this.ubication = ubication;
    }
    
    /**
     * Devuelve la ruta a la imagen del evento
     * 
     * @return imagen Ruta de la imagen
     */
    public String getImagen() {
        return imagen;
    }

    /**
     * Establece la imagen asociada al evento
     * 
     * @param imagen Ruta a la imagen del evento
     */
    public void setImagen(String imagen) {
        this.imagen = imagen;
    }
    
    /**
     * Tras la subida de la imagen establece una por defecto
     */
    @PostLoad
    private void asignarImagenPorDefecto() {
        if (this.imagen == null || this.imagen.isBlank()) {
            this.imagen = "default-event.png";
        }
    }
    
    /**
     * Devuelve el usuario dueño del evento
     * 
     * @return owner Usuario del dueño del evento
     */
    public User getOwner() {
        return owner;
    }
    
    /**
     * Devuelve el aforo máximo del evento
     * 
     * @return Aforo del evento
     */
    public int getMaxCapacity() {
        return maxCapacity;
    }

    /**
     * Establece el aforo máximo del evento
     * 
     * @param maxCapacity Aforo del evento
     */
    public void setMaxCapacity(int maxCapacity) {
        this.maxCapacity = maxCapacity;
    }

    /**
     * Devuelve el número actual de participantes
     * 
     * @return Número actual de participantes
     */
    public int getNumParticipants() {
        return numParticipants;
    }
    
    /**
     * Logica para añadir un participante
     */
    public boolean increaseNumParticipant() {
    	if(numParticipants < maxCapacity) {
    		numParticipants++;
    		return true;
    	}
    	
    	return false;
    }
    
    /**
     * Logica para eliminar un participante
     */
    public boolean decreaseNumParticipant() {
    	if(numParticipants > 0) {
    		numParticipants--;
    		return true;
    	}
    	
    	return false;
    }
    
    /**
     * Establece el usuario dueño del evento
     * 
     * @param owner Usuario dueño del evento
     */
    public void setOwner(User owner) {
        this.owner = owner;
    }

    /**
     * Establece las categorías del evento
     * 
     * @param categories Categorías del evento
     */
	public void setCategories(List<EventCategory> categories) {
		this.categories = categories;
		
	}
	
	/**
     * Devuelve las categorías del evento
     * 
     * @return categorías a las que pertenece el evento
     */
	public List<EventCategory> getCategories() {
		return categories;
	}
	
	/**
     * Añade una actividad
     * 
     * @param owner Usuario dueño del evento
     */
	public void setActivities(List<Activity> activities) {
	    this.activities = activities;
	}
	
	/**
     * Devuelve las actividades del evento
     * 
     * @return lita de actividades asociadas al evento
     */
	public List<Activity> getActivities() {
	    return activities;
	}

	/**
	 * Añade una actividad del evento
	 * 
	 * @param actividad Actividad que se quiere añadir
	 */
	public void addActividad(Activity actividad) {
		activities.add(actividad);
	    actividad.setEvent(this);
	}

	/**
	 * Elimina una actividad del evento
	 * 
	 * @param actividad Actividad que se desea eliminar
	 */
	public void removeActividad(Activity actividad) {
		activities.remove(actividad);
	    actividad.setEvent(null);
	}
	
	/**
	 * Devuelve la lista de los usuarios que participan en el evento
	 * 
	 * @return List<User> Lista de participantes del evento
	 */
	public List<User> getParticipants() {
	    return participants;
	}

	/**
	 * Establece la lista de los usuarios que participan en el evento
	 * 
	 * @param participants Lista de participantes
	 */
	public void setParticipants(List<User> participants) {
	    this.participants = participants;
	}
	
	/**
	 * Añade un usuario a un evento en el que no participaba previamente
	 * 
	 * @param user Usuario que se inscribe al evento
	 * @return boolean True si se pudo añadir y false en caso contrario
	 */
	public boolean addParticipant(User user) {
	    if (!participants.contains(user)) {
	    	// Añade el participante
	    	participants.add(user);
	    	
	    	// Incrementa el contador de participantes
	    	this.increaseNumParticipant();
	    	
	    	return true;
	    }
	    
	    return false;
	}

	/**
	 * Elimina un usuario de un evento en el que participaba
	 * 
	 * @param user Usuario que se desea desapuntar del evento
	 * @return boolean True si se pudo eliminar y false en caso contrario
	 */
	public boolean removeParticipant(User user) {
		if (participants.contains(user)) {
			// Elimina el usuario del evento
			participants.remove(user);
			
			// Decrementa el contador de participantes
			this.decreaseNumParticipant();
		}

		return false;
	}
}
