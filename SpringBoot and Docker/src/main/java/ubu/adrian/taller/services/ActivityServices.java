package ubu.adrian.taller.services;

import java.util.List;

import org.springframework.stereotype.Service;

import ubu.adrian.taller.model.Activity;
import ubu.adrian.taller.repository.ActivityRepository;

/**
 * Servicio para la gestión de actividades
 */
@Service
public interface ActivityServices {
	/**
     * Devuelve una actividad dado su ID
     * 
     * @param id ID de la actividad
     * @return Activity asociado al ID
     */
	public Activity findById(long id);
	
	/**
	 * Devuelve una lista de todas las actividades
	 * 
	 * @return List<Activity> Lista de todos los eventos
	 */
	public List <Activity> findAll();
	
	/**
	 * Guarda la actividad en la base de datos
	 * 
	 * @param activity Actividad que se quiere guardar
	 */
	public void saveActivity(Activity activity);
	
	/**
	 * Guarda la actividad en la base de datos
	 * 
	 * @param activity Actividad que se quiere guardar
	 */
	public void deleteActivity(Activity activity);
}
