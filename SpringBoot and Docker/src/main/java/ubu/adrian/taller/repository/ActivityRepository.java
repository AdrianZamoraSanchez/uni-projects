package ubu.adrian.taller.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import ubu.adrian.taller.model.Activity;
import ubu.adrian.taller.model.Event;

/**
 * Repositorio para las actividades
 */
@Repository
public interface ActivityRepository extends JpaRepository<Activity, Long> {
	
	/**
	 * Devuelve una actividad dado su ID
	 * 
	 * @param activityID ID por el que se busca
	 * @return Actividad asociada al ID
	 */
	public Activity findById(long activityID);
}
