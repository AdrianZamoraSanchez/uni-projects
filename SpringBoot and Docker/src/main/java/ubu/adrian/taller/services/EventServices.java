package ubu.adrian.taller.services;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import ubu.adrian.taller.model.Categories;
import ubu.adrian.taller.model.Event;
import ubu.adrian.taller.model.User;
import ubu.adrian.taller.repository.EventRepository;

/**
 * Servicio para la gestión de eventos
 */
public interface EventServices {
	/**
     * Devuelve el evento dado su ID
     * 
     * @param id ID del evento
     * @return Evento asociado al ID
     */
	public Event findById(long id);
	
	/**
	 * Devuelve una lista de todos los eventos
	 * 
	 * @return List<eventos> Lista de todos los eventos
	 */
	public List <Event> findAll();
	
	/**
     * Devuelve los eventos tras aplicarle unos filtros
     * 
     * @param category Categorías que se quiere obtener
     * @param capacity Disponibilidad de plazas
     * @return Eventos que cumplen con los filtros
     */
	public List<Event> findByFilters(List<Categories> categories, String capacity);
	
	/**
	 * Inscribe un usuario al evento
	 * 
	 * @param event Evento al que se añade el usuario
	 * @param user Usuario que se añade al evento
	 */
	public void addParticipantToEvent(Event event, User user);

	/**
	 * Da de baja un usuario al evento
	 * 
	 * @param event Evento al que se desinscribe el usuario
	 * @param user Usuario que se desinscribe del evento
	 */
	public void removeParticipantToEvent(Event event, User user);
	
	/**
	 * Guarda al evento en la base de datos
	 * 
	 * @param event Evento que se quiere guardar
	 */
	public void saveEvent(Event event);
	
	/**
	 * Elimina el evento dado su ID
	 * 
	 * @param id ID del evento que se desea borrar
	 */
	public void deleteById(long id);
	
	/**
	 * Devuelve los eventos que ha creado un usuario
	 * 
	 * @param user Usuario que ha creado los eventos
	 * @return Eventos creados por dicho usuario
	 */
	public List<Event> findByOwner(User owner);
	
	/**
	 * Busca los eventos comunes a un participante
	 * 
	 * @param user Participante cuyos eventos se quiere encontrar
	 * @return Lista de eventos en los que participa el usuario
	 */
	public List<Event> findEventsByParticipant(User user);
}
