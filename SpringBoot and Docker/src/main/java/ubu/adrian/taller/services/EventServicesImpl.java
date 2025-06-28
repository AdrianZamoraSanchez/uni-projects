package ubu.adrian.taller.services;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import ubu.adrian.taller.model.Categories;
import ubu.adrian.taller.model.Event;
import ubu.adrian.taller.model.User;
import ubu.adrian.taller.repository.EventRepository;

@Service
public class EventServicesImpl implements EventServices {
	// Repositorio con los datos de los eventos
	@Autowired
	private EventRepository eventRepository;

    // Inyección del repositorio a través del constructor
    public EventServicesImpl(EventRepository eventRepository) {
        this.eventRepository = eventRepository;
    }
    
    /**
     * Devuelve el evento dado su ID
     * 
     * @param id ID del evento
     * @return Evento asociado al ID
     */
    public Event findById(long id) {
    	return eventRepository.findById(id);
    };
    
    /**
	 * Devuelve una lista de todos los eventos
	 * 
	 * @return List<eventos> Lista de todos los eventos
	 */
    public List <Event> findAll() {
        return eventRepository.findAll();
    }
    
    /**
	 * Guarda al evento en la base de datos
	 * 
	 * @param event Evento que se quiere guardar
	 */
    public void saveEvent(Event event) {
    	eventRepository.save(event);
    }

    /**
     * Devuelve los eventos tras aplicarle unos filtros
     * 
     * @param category Categorías que se quiere obtener
     * @param capacity Disponibilidad de plazas
     * @return Eventos que cumplen con los filtros
     */
	public List<Event> findByFilters(List<Categories> categories, String capacity) {
		if (categories == null && capacity == null) {
	        return eventRepository.findAll();
	    }

		// Comprueba las diferentes combinaciones de filtros
	    if (categories != null && "FULL".equals(capacity)) {
	        return eventRepository.findByCategoriesAndFull(categories);
	    } else if (categories != null && "AVAILABLE".equals(capacity)) {
	        return eventRepository.findByCategoriesAndAvailable(categories);
	    } else if (categories != null) {
	        return eventRepository.findByCategories(categories);
	    } else if ("FULL".equals(capacity)) {
	        return eventRepository.findFull();
	    } else if ("AVAILABLE".equals(capacity)) {
	        return eventRepository.findAvailable();
	    }

	    return eventRepository.findAll();
	};
	
	/**
	 * Inscribe un usuario al evento
	 * 
	 * @param event Evento al que se añade el usuario
	 * @param user Usuario que se añade al evento
	 */
	public void addParticipantToEvent(Event event, User user) {
	    event.getParticipants().add(user);
	    event.increaseNumParticipant();
	    eventRepository.save(event);
	}
	
	/**
	 * Da de baja un usuario al evento
	 * 
	 * @param event Evento al que se desinscribe el usuario
	 * @param user Usuario que se desinscribe del evento
	 */
	public void removeParticipantToEvent(Event event, User user) {
	    event.getParticipants().remove(user);
	    event.decreaseNumParticipant();
	    eventRepository.save(event);
	}
	
	/**
	 * Elimina el evento dado su ID
	 * 
	 * @param id ID del evento que se desea borrar
	 */
	public void deleteById(long id) {
	    eventRepository.deleteById(id);
	}
	
	/**
	 * Devuelve los eventos que ha creado un usuario
	 * 
	 * @param user Usuario que ha creado los eventos
	 * @return Eventos creados por dicho usuario
	 */
	public List<Event> findByOwner(User owner) {
	    return eventRepository.findByOwner(owner);
	}
	
	/**
	 * Busca los eventos comunes a un participante
	 * 
	 * @param user Participante cuyos eventos se quiere encontrar
	 * @return Lista de eventos en los que participa el usuario
	 */
	public List<Event> findEventsByParticipant(User user) {
	    return eventRepository.findByParticipantsContaining(user);
	}
}
