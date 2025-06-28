package ubu.adrian.taller.services;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import ubu.adrian.taller.model.Activity;
import ubu.adrian.taller.repository.ActivityRepository;

/**
 * Implementación de la interfaz ActivityServices
 */
@Service
public class ActivityServicesImpl implements ActivityServices{
	// Repositorio de actividades
	@Autowired
	ActivityRepository activityRepository;
	
	// Inyección del repositorio a través del constructor
    public ActivityServicesImpl(ActivityRepository activityRepository) {
        this.activityRepository = activityRepository;
    }
    
    /**
     * Devuelve una actividad dado su ID
     * 
     * @param id ID de la actividad
     * @return Activity asociado al ID
     */
    public Activity findById(long id) {
    	return activityRepository.findById(id);
    };
    
    /**
	 * Devuelve una lista de todas las actividades
	 * 
	 * @return List<Activity> Lista de todos los eventos
	 */
    public List <Activity> findAll() {
        return activityRepository.findAll();
    }
    
    /**
	 * Guarda la actividad en la base de datos
	 * 
	 * @param activity Actividad que se quiere guardar
	 */
    public void saveActivity(Activity activity) {
		activityRepository.save(activity);
	};
	
	/**
	 * Guarda la actividad en la base de datos
	 * 
	 * @param activity Actividad que se quiere guardar
	 */
    public void deleteActivity(Activity activity) {
		activityRepository.delete(activity);
	};
}
