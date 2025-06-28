package ubu.adrian.taller.controller;


import java.util.List;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.ui.Model;

import ubu.adrian.taller.dto.UserRegisterDTO;
import ubu.adrian.taller.dto.UserMapper;
import ubu.adrian.taller.model.Event;
import ubu.adrian.taller.model.User;
import ubu.adrian.taller.model.UserRol;
import ubu.adrian.taller.repository.UserRepository;
import ubu.adrian.taller.services.EventServicesImpl;
import ubu.adrian.taller.services.UserServices;
import ubu.adrian.taller.services.UserServicesImpl;

/**
 * Controlador del panel de administración
 * 
 * @author Adrián Zamora Sánchez (azs1004@alu.ubu.es)
 */
@Controller
public class AdminController {
	// Servicio de usuarios
	@Autowired
	private UserServicesImpl userServices;
	
	// Servicio de eventos
	@Autowired
	private EventServicesImpl eventServices;
	
	// Servicio de encriptación de contraseñas
	@Autowired
    private PasswordEncoder passwordEncoder;
	
	// Mapper de UserDTO a User
	@Autowired
	private UserMapper userMapper;

    /**
	 * Gestiona las solicitudes de la ruta /admin
	 * 
	 * @param model Modelo donde se insertan los datos
	 * @return panel del administrador
	 */
    @GetMapping("/user-list")
    public String showUserList(Model model) {
    	// Lista de usuarios
    	List<UserRegisterDTO> userDTOs = userServices.getAllUsers()
                .stream()
                .map(userMapper::toDTO)
                .toList();
    	
    	// Se añaden al modelo
    	model.addAttribute("listUsers", userDTOs);
        
        return "admin";
    }
    
    /**
     * Gestiona la eliminación de usuarios
     * 
     * @param id identificador del usuario que se quiere eliminar
     * @return la lista de usuarios actualizada
     */
    @PostMapping("/remove")
    public String removeUser(@RequestParam("id") long id) {
    	User user = userServices.findById(id);
    	
    	List<Event> events = null;
    	
    	// Comprueba si el usuario es un organizador
    	if (user.getRol() == UserRol.ORGANIZADOR) {
            List<Event> eventsOwned = eventServices.findByOwner(user);
            
            // Elimina sus eventos
            for (Event event : eventsOwned) {
                eventServices.deleteById(event.getId());
            }
        }
    	
    	 // Si es participante, elimínalo de los eventos donde participa
        List<Event> eventsAsParticipant = eventServices.findEventsByParticipant(user);
        for (Event event : eventsAsParticipant) {
        	// Lo elimina
            event.removeParticipant(user);
            
            // Guarda el cambio
            eventServices.saveEvent(event);     
        }
        
    	// Elimina por id
        userServices.deleteById(id);
        return "redirect:/user-list";
    }
    
    /**
	 * Gestiona las solicitudes de la ruta /updateUserData
	 * 
	 * @param id identificador del usuario que se quiere actualizar
	 * @param model Modelo donde se insertan los datos
	 * @return panel de actualización de usuarios
	 */
    @GetMapping("/update-user-data")
    public String updateUserData(@RequestParam Long id, Model model) {
    	User user = userServices.findById(id);
    	
        model.addAttribute("user", user);
        model.addAttribute("rolList", UserRol.values());
    	
        return "updateUserData";
    }
    
    /**
	 * Gestiona las actualizaciones de datos de usuarios
	 * 
	 * @param updatedUserDTO Datos actualizados del usuario
	 * @param model Modelo donde se insertan los datos
	 * @return lista de usuarios
	 */
    @PostMapping("/update-user")
    public String updateUser(@ModelAttribute("user") UserRegisterDTO updatedUserDTO, Model model) {
        // Busca al usuario (el cual puede no estar)
    	User existingUser = userServices.findById(updatedUserDTO.getId());
        
    	// Comprueba si el usuario esta presente
    	if (existingUser != null) {
    		// Toma los datos del usuario
    		existingUser.setUsername(updatedUserDTO.getUsername());

            // Encripta la nueva contraseña
            String passwordEncriptada = passwordEncoder.encode(updatedUserDTO.getPassword());
            existingUser.setPassword(passwordEncriptada);

            existingUser.setRol(updatedUserDTO.getRol());
            
            // Lo vuelve a guardar (reescribe sus datos)
            userServices.saveUser(existingUser);
        } else {
        	// Devuelve mensaje de error
            model.addAttribute("error", "No se encontró un usuario con el ID especificado.");
        }

    	return "redirect:/user-list";
    }
}