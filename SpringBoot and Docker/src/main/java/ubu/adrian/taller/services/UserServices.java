package ubu.adrian.taller.services;

import java.util.List;

import ubu.adrian.taller.model.User;

/**
 * Servicios que se implementan sobre las entidades de usuarios
 */
public interface UserServices {
	/**
	 * Devuelve una lista de todos los usuarios
	 * 
	 * @return List<User> lista de todos los usuario encontrados
	 */
    List <User> getAllUsers();
    
    /**
	 * Guarda al usuario en la base de datos
	 * 
	 * @param user Usuario que se quiere guardar
	 */
    void saveUser(User user);
    
    /**
	 * Devuelve un usuario especificado
	 * 
	 * @param id Identificador del usuario que se busca
	 */
    User findById(long id);
    
    /**
	 * Elimina al usuario especificado
	 * 
	 * @param id Identificador del usuario que se desea eliminar
	 */
    void deleteById(long id);

    /**
	 * Devuelve un usuario especificado
	 * 
	 * @param username Nombre del usuario que se busca
	 */
	User findByUsername(String username);
}
