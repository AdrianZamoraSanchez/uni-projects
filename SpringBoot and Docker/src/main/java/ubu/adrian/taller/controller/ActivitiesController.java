package ubu.adrian.taller.controller;


import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

import ubu.adrian.taller.dto.NewActivityDTO;
import ubu.adrian.taller.model.Activity;
import ubu.adrian.taller.model.Event;
import ubu.adrian.taller.model.User;
import ubu.adrian.taller.model.UserRol;
import ubu.adrian.taller.services.ActivityServicesImpl;
import ubu.adrian.taller.services.EventServicesImpl;
import ubu.adrian.taller.services.UserServicesImpl;

/**
 * Controlador la gestión de actividades de eventos
 * 
 * @author Adrián Zamora Sánchez (azs1004@alu.ubu.es)
 */
@Controller
public class ActivitiesController {
	
	// Servicios utilizados
	@Autowired
	EventServicesImpl eventServices;
	
	@Autowired
	ActivityServicesImpl activityServices;
	
	@Autowired
	UserServicesImpl userServices;
	
	/**
	 * Constructor del controlador de actividades
	 * 
	 * @param eventServices servicio de eventos
	 * @param activityServices servicio de actividades
	 * @param userServices servicio de usuarios
	 */
    public ActivitiesController(EventServicesImpl eventServices, ActivityServicesImpl activityServices, UserServicesImpl userServices) {
        this.eventServices = eventServices;
        this.activityServices = activityServices;
        this.userServices = userServices;
    }
	
	/**
	 * Gestiona la creacion de una actividad para un evento
	 * 
	 * @param eventId ID del evento
	 * @param model Modelo de la vista
	 * @return lista de actividades
	 */
	@GetMapping("/activity/manage/{eventId}")
	public String activityList(@PathVariable long eventId, Model model) {
		Event event = eventServices.findById(eventId);
	    
		if (event == null) {
	        return "redirect:/event/list";
	    }

	    List<Activity> activities = event.getActivities();

	    // Se agregan los datos al modelo
	    model.addAttribute("event", event);
	    model.addAttribute("activities", activities);

	    return "activityList";
	}
	
	/**
	 * Elimina una actividad
	 * 
	 * @param eventId ID del evento 
	 * @param activityId ID de la actividad
	 * @return lista de actividades
	 */
	@GetMapping("/activity/delete/{id}")
	public String deleteActivity(@PathVariable("id") long activityId, @RequestParam("eventId") long eventId) {
	    Activity activity = activityServices.findById(activityId);

	    if (activity != null) {
	        activityServices.deleteActivity(activity);
	    }

	    // Redirige de nuevo a la gestión de actividades del evento
	    return "redirect:/activity/manage/" + eventId;
	}
	
	/**
	 * Gestiona la creacion de una actividad para un evento
	 * 
	 * @param eventId ID del evento
	 * @param model Modelos de la vista
	 * @return formato de creacion de actividades
	 */
	@GetMapping("/activity/create/{eventId}")
	public String createActivityForm(@PathVariable long eventId, Model model) {
	    Event event = eventServices.findById(eventId);
	    if (event == null) {
	        return "redirect:/event/list"; // O error
	    }

	    Activity activity = new Activity();
	    activity.setEvent(event);

	    model.addAttribute("activity", activity);
	    model.addAttribute("event", event);
	    return "createActivity";
	}
	
	/**
	 * Gestiona la creacion de una actividad para un evento
	 * 
	 * @param eventId ID del evento
	 * @param model Modelos de la vista
	 * @return página de información del evento
	 */
	@PostMapping("/activity/save-activity")
	public String saveActivity(@ModelAttribute Activity activity, 
			@RequestParam("eventId") long eventId,
			Authentication authentication) {
		
	    // Nos aseguramos de que el evento esté bien enlazado
	    Event event = eventServices.findById(eventId);
	    
	    // Validar autoría del evento
	    User user = userServices.findByUsername(authentication.getName());
	    if (event == null || !event.getOwner().equals(user)) {
	        return "redirect:/login";
	    }
	    
	    activity.setEvent(event);

	    activityServices.saveActivity(activity);

	    return "redirect:/event/info/" + event.getId();
	}
	
	/**
     * Edita los datos de un evento existente
     * 
     * @param eventId ID del evento
	 * @param model Modelos de la vista
     * @param authentication token de autenticación del usuario
     * @return 
     */
    @GetMapping("/activity/edit/{id}")
    public String editActivity(@PathVariable long id, Model model, Authentication authentication) {
    	Activity activity = activityServices.findById(id);
    	
    	Event event = activity.getEvent();
    	
        // El usuario debe ser autenticado como owner del evento
        String username = authentication.getName();
        if (!event.getOwner().getUsername().equals(username)) {
            return "redirect:/login";
        }

        // Crear DTO con los valores del evento
        NewActivityDTO dto = new NewActivityDTO();
        model.addAttribute("id", id);
        dto.setName(activity.getName());
        dto.setDescription(activity.getDescription());
        dto.setStartTime(activity.getStartTime());
        dto.setEndTime(activity.getEndTime());

        // Convertir las categorías del evento a lista de Strings
        
        model.addAttribute("activityDTO", dto);
        return "updateActivity";
    }
    
    /**
     * Actualiza la información de una actividad
     * 
     * @param id ID del evento que se quiere actualizar
     * @param updatedActivity Objeto que contiene los datos actualizados
     * @param authentication token de autenticación del usuario
     * @return Página de información del evento actualizado
     */
	@PostMapping("/update-activity/{id}")
	public String updateActivity(@PathVariable long id, 
			@ModelAttribute NewActivityDTO updatedActivity, 
			Authentication authentication) {
		
	    Activity existing = activityServices.findById(id);
	    
	    Event event = existing.getEvent();
	    // Comprueba que el dueño del evento es el que lo está editando
        User user = null;
        if (authentication != null && authentication.isAuthenticated()) {
            user = userServices.findByUsername(authentication.getName());
        }

        // Comprueba si es el dueño para mostrar datos de organizador del evento
        if (!event.getOwner().equals(user) && user.getRol() != UserRol.ORGANIZADOR) {
            return "redirect:/login";
        }
	    
	    // Datos actualizados
	    existing.setName(updatedActivity.getName());
	    existing.setDescription(updatedActivity.getDescription());
	    existing.setStartTime(updatedActivity.getStartTime());
	    existing.setEndTime(updatedActivity.getEndTime());
	    
        // Actualiza los datos de la actividad
	    activityServices.saveActivity(existing);
	    return "redirect:/event/info/" + event.getId();
	}
}