package ubu.adrian.taller.repository;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import ubu.adrian.taller.model.Categories;
import ubu.adrian.taller.model.Event;
import ubu.adrian.taller.model.User;

/**
 * Repositorio de la entidad User
 */
@Repository
public interface EventRepository extends JpaRepository<Event, Long> {
	
	/**
     * Busca un evento por su ID
     * 
     * @param eventID ID del evento
     * @return Event encontrado o null
     */
    public Event findById(long eventID);
    
    /**
     * Busca los eventos que pertenecen a un usuario
     * 
     * @param owner Usuario que ha creado eventos
     * @return Eventos que pertenecen a ese owner
     */
    List<Event> findByOwner(User owner);

    /**
     * Busca los eventos en los que está inscrito un usuario
     * 
     * @param participant Usuario participante en algún evento
     * @return Eventos en los que participa el usuario
     */
    List<Event> findByParticipantsContaining(User participant);

    /**
     * Devuelve los eventos que cumplen con las categorías buscadas y estan llenos
     * 
     * @param categories Categorías por las que se busca
     * @return Lista<Event> Lista de eventos que cumplen las condiciones
     */
    @Query("SELECT DISTINCT e FROM Event e " +
    	       "JOIN e.categories c " +
    	       "WHERE c.category IN :categories " +
    	       "AND e.numParticipants >= e.maxCapacity")
    	List<Event> findByCategoriesAndFull(@Param("categories") List<Categories> categories);

    /**
     * Devuelve los eventos que cumplen con las categorías buscadas y no estan llenos
     * 
     * @param categories Categorías por las que se busca
     * @return Lista<Event> Lista de eventos que cumplen las condiciones
     */
    @Query("SELECT DISTINCT e FROM Event e " +
    	       "JOIN e.categories c " +
    	       "WHERE c.category IN :categories " +
    	       "AND e.numParticipants < e.maxCapacity")
	List<Event> findByCategoriesAndAvailable(@Param("categories") List<Categories> categories);


    /**
     * Devuelve los eventos que están llenos
     * 
     * @return Lista<Event> Lista de eventos que cumplen las condiciones
     */
     @Query("SELECT e FROM Event e " +
            "WHERE e.numParticipants >= e.maxCapacity")
     List<Event> findFull();
  

     /**
      * Devuelve los eventos que no están llenos
      * 
      * @return Lista<Event> Lista de eventos que cumplen las condiciones
      */
     @Query("SELECT e FROM Event e " +
            "WHERE e.numParticipants < e.maxCapacity")
     List<Event> findAvailable();

     /**
      * Devuelve los eventos de una determinada categoría
      * 
      * @return Lista<Event> Lista de eventos que cumplen las condiciones
      */
     @Query("SELECT DISTINCT e FROM Event e " +
    	       "JOIN e.categories c " +
    	       "WHERE c.category IN :categories")
	List<Event> findByCategories(@Param("categories") List<Categories> categories);

}