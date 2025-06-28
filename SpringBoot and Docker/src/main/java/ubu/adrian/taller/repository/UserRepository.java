package ubu.adrian.taller.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import ubu.adrian.taller.model.User;

/**
 * Repositorio de la entidad User
 */
@Repository
public interface UserRepository extends JpaRepository<User, Long> {
	
	/**
     * Busca un usuario por su nombre de usuario
     * 
     * @param username Nombre de usuario a buscar
     * @return Usuario encontrado o null
     */
    User findByUsername(String username);
}