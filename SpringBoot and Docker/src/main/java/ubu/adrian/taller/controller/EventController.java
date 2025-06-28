package ubu.adrian.taller.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import ubu.adrian.taller.services.EventServicesImpl;
import ubu.adrian.taller.services.UserServicesImpl;
import ubu.adrian.taller.dto.NewEventDTO;
import ubu.adrian.taller.model.Categories;
import ubu.adrian.taller.model.Event;
import ubu.adrian.taller.model.EventCategory;
import ubu.adrian.taller.model.User;
import ubu.adrian.taller.model.UserRol;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;
/**
 * Controlador de las páginas relacionadas con la gestión de usuarios
 * 
 * @author Adrián Zamora Sánchez (azs1004@alu.ubu.es)
 */
@Controller
public class EventController {
	// Servicio de eventos
	@Autowired
	private EventServicesImpl eventServices;
	
	// Servicio de usuarios
	@Autowired
	private UserServicesImpl userServices;
	
    /**
     * Visualiza los eventos a los que está inscrito un usuario
     * o los que pertenecen a un organizador según sea el rol de usuario
     * 
     * @param model Modelo de la vista
     * @param authentication token de autenticación del usuario
     * @return lista de eventos del usuario
     */
    @GetMapping("/event/my-events")
    public String myEvents(Model model, Authentication authentication) {
        User user = userServices.findByUsername(authentication.getName());

        List<Event> events;

        if (user.getRol() == UserRol.ORGANIZADOR) {
            events = eventServices.findByOwner(user);
        } else if (user.getRol() == UserRol.PARTICIPANTE) {
            events = eventServices.findEventsByParticipant(user);
        } else {
            events = new ArrayList<>();
        }

        model.addAttribute("eventList", events);
        return "myEvents";
    }
    
    /**
     * Elimina un evento
     * 
     * @param id ID del evento
     * @return Al home o a la lista de eventos
     */
    @PostMapping("/event/delete/{id}")
    public String deleteEvent(@PathVariable Long id, 
    		Authentication authentication, 
    		RedirectAttributes redirectAttributes) {
        Event event = eventServices.findById(id);
        User user = userServices.findByUsername(authentication.getName());

        // Autorización: solo el creador o un ADMIN puede borrar
        if (event.getOwner().equals(user) || user.getRol() == UserRol.ADMIN) {
            eventServices.deleteById(id);
            
            // Elimina la imagen del evento si no es la por defecto
            if (event.getImagen() != null && !event.getImagen().equals("default-event.png")) {
            	// Ruta dentro del contenedor
                Path uploadPath = Paths.get(uploadDir + "/img/event/");
            	Path imagePath = uploadPath.resolve(event.getImagen());
            	
                try {
                    Files.deleteIfExists(imagePath);
                } catch (IOException e) {
                    System.out.println("No se pudo eliminar la imagen anterior: " + e.getMessage());
                }
            }
            
            return "redirect:/";
        }

        return "redirect:/event/my-events";
    }
    
    /**
     * Añade un participante al evento
     * 
     * @param id ID del evento
     * @param authentication
     * @return Página de login en caso de necesitar login o información del evento
     */
    @PostMapping("/event/join/{id}")
    public String joinEvent(@PathVariable long id, 
    		Authentication authentication, 
    		RedirectAttributes redirectAttributes) {
    	
        // Se comprueba que el usuario está autenticado
    	if (authentication == null || !authentication.isAuthenticated()) {
            return "redirect:/login?error=Debes+iniciar+sesión+para+participar+en+un+evento";
        }

    	// Se obtienen los datos del usuario y evento
        String username = authentication.getName();
        User user = userServices.findByUsername(username);
        Event event = eventServices.findById(id);

        // Comprueba si el usuario puede apuntarse al evento (no debe pertenecer previamente)
        if (event.getParticipants().contains(user)) {
        	return "redirect:/login?error=No+puedes+inscribirte+a+un+evento+alque+ya+perteneces";
        } else {
            eventServices.addParticipantToEvent(event, user);
        }

        return "redirect:/event/info/" + id;
    }
    
    /**
     * Desinscribe un participante del evento
     * 
     * @param eventId ID del evento
     * @param userId ID del usuario
     * @param authentication Token de autenticación del usuario
     * @return Página de eventos
     */
    @PostMapping("/event/{eventId}/remove-user/{userId}")
    public String removeParticipant(@PathVariable long eventId,
                                    @PathVariable long userId,
                                    Authentication authentication) {

        User currentUser = userServices.findByUsername(authentication.getName());
        Event event = eventServices.findById(eventId);

        // Solo el dueño, que es organizador puede eliminar participantes
        if (!event.getOwner().equals(currentUser) && currentUser.getRol() != UserRol.ORGANIZADOR) {
            return "redirect:/login?error=Debes+iniciar+sesión+como+organizador+para+editar+el+evento";
        }

        // Toma el usuario que se quiere eliminar y lo borra del evento
        User userToRemove = userServices.findById(userId);
        eventServices.removeParticipantToEvent(event, userToRemove);

        return "redirect:/event/info/" + eventId;
    }
    
    /**
     * Desinscribe un participante del evento 
     * para uso de organizadores
     * 
     * @param id ID del evento
     * @param authentication Token de autenticación del usuario
     * @return Página de eventos
     */
    @PostMapping("/event/leave/{id}")
    public String leaveEvent(@PathVariable long id, 
    		Authentication authentication) {
        // Se comprueba que el usuario está autenticado
    	if (authentication == null || !authentication.isAuthenticated()) {
            return "redirect:/login?error=Debes+iniciar+sesión+para+dejar+un+evento";
        }

    	// Se obtienen los datos del usuario y evento
        String username = authentication.getName();
        User user = userServices.findByUsername(username);
        Event event = eventServices.findById(id);

        // Comprueba si el usuario puede salir del evento (debe pertenecer previamente)
        if (!event.getParticipants().contains(user)) {
        	return "redirect:/login?error=Debes+pertenecer+al+evento+antes+de+salir";
        } else {
            eventServices.removeParticipantToEvent(event, user);
        }

        return "redirect:/event/my-events";
    }
    
    /**
     * Gestiona la página de visionado de un evento en la ruta
     * /event/info/id_evento
     * 
     * @param id ID del evento
     * @param model Modelo de la vista
     * @param authentication Token de autenticación del usuario
     * @return página de información del evento
     */
    @GetMapping("/event/info/{id}")
    public String eventInfo(@PathVariable long id, 
    		Model model, 
    		Authentication authentication){
    	// Evento y dueño
        Event event = eventServices.findById(id);
        boolean owner = false;
        
        // Si está logueado se obtiene su info, sino queda null
        User user = null;
        if (authentication != null && authentication.isAuthenticated()) {
            user = userServices.findByUsername(authentication.getName());
        }

        // Comprueba si es el dueño para mostrar datos de organizador del evento
        if (event.getOwner().equals(user) && user.getRol() == UserRol.ORGANIZADOR) {
            owner = true;
        }

        // Se añade información al modelo
        model.addAttribute("owner", owner);
        model.addAttribute("event", event);
        model.addAttribute("user", user);
    	
    	return "eventInfo";
	}
    
    /**
	 * Gestiona las solicitudes de la ruta /event/search
	 * 
	 * @param categories Lista de categorias a biscar
	 * @param capacity Si esta lleno o no el evento
	 * @param model Modelo de la vista
	 * @return página de creación de eventos
	 */
    @GetMapping("/event/search")
    public String eventList(
    		@RequestParam(required = false) List<Categories> categories,
            @RequestParam(required = false) String capacity,
            Model model
    ) {
        List<Event> eventList;

        // Lógica de filtrado
        if (categories == null && (capacity == null || capacity.isEmpty())) {
            eventList = eventServices.findAll();
        } else {
            // Filtrar según los parámetros
        	eventList = eventServices.findByFilters(categories, capacity);
        }

        model.addAttribute("eventList", eventList);
        model.addAttribute("categories", Categories.values());

        // Para mantener el estado del filtro en la vista
        model.addAttribute("selectedCategories", categories);
        model.addAttribute("selectedCapacity", capacity);

        return "eventList";
    }
    
    /**
	 * Gestiona las solicitudes de la ruta /event/filter
	 * 
	 * @param model Modelo de la vista
	 * @return página de creación de eventos
	 */
    @GetMapping("/event/filter")
    public String eventFilter(Model model) {
        // Obtiene y añade la lista de eventos al modelo
        List<Event> eventList = eventServices.findAll();
        model.addAttribute("eventList", eventList);
        
        // Se añaden las categorías disponibles
        model.addAttribute("categories", Categories.values());
        
        return "eventList";
    }

	/**
	 * Gestiona las solicitudes de la ruta /event/create
	 * 
	 * @param model Modelo de la vista
	 * @return página de creación de eventos
	 */
    @GetMapping("/event/create")
    public String createEvent(Model model) {
        model.addAttribute("categories", Categories.values());
        return "createEvent";
    }
    
    // Ubicación para ruta de almacenamiento de imágenes
    @Value("${app.upload.dir:/app/uploads}")
    private String uploadDir;
    
    /**
     * Gestiona la creación de un nuevo evento
     * 
     * @param newEventDTO DTO del evento que se crea
     * @return Redirección a la lista de eventos
     */
    @PostMapping("/new-event")
    public String newEvent(@ModelAttribute NewEventDTO newEventDTO) throws IOException {
        
    	// Obtener usuario autenticado
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String username = authentication.getName();
        
        // Buscar usuario en BD
        User owner = userServices.findByUsername(username);
    	
        // Crear nueva entidad Event
        Event event = new Event();
        event.setTitle(newEventDTO.getTitle());
        event.setDescription(newEventDTO.getDescription());
        event.setStartDate(newEventDTO.getStartDate());
        event.setEndDate(newEventDTO.getEndDate());
        event.setUbication(newEventDTO.getUbication());
        event.setOwner(owner);
        event.setMaxCapacity(newEventDTO.getMaxCapacity());
        
        // Manejar la imagen
        if (!newEventDTO.getImagen().isEmpty()) {
            String nombreArchivo = UUID.randomUUID() + "_" + newEventDTO.getImagen().getOriginalFilename();
            
            // Ruta dentro del contenedor
            Path uploadPath = Paths.get(uploadDir + "/img/event/");
            Files.createDirectories(uploadPath);
            
            Path filePath = uploadPath.resolve(nombreArchivo);
            Files.copy(newEventDTO.getImagen().getInputStream(), filePath, StandardCopyOption.REPLACE_EXISTING);
            
            event.setImagen(nombreArchivo); // Solo nombre del archivo
        } else {
            event.setImagen("default-event.png"); // Imagen por defecto
        }
        
        // Convertir Strings a Categories
        List<Categories> categoryEnums = new ArrayList<>();
        if(newEventDTO.getCategories() != null) {
            for(String catStr : newEventDTO.getCategories()) {
                try {
                    Categories catEnum = Categories.valueOf(catStr);
                    categoryEnums.add(catEnum);
                } catch (IllegalArgumentException e) {
                    System.out.println("Categoria no valida: " + catStr);
                }
            }
        }
        
        // Crear EventCategory para cada enum
        List<EventCategory> eventCategories = categoryEnums.stream()
            .map(catEnum -> new EventCategory(catEnum, event))
            .collect(Collectors.toList());
        
        event.getCategories().addAll(eventCategories);
        
        // Guardar el evento (se guardan en cascada las categorías)
        eventServices.saveEvent(event);
        
        return "redirect:/event/search";
    }
    
    /**
     * Edita los datos de un evento existente
     * 
     * @param id ID del evento que se edita
     * @param model Modelo de la vista
     * @param authentication Autenticador de usuarios
     * @return Formulario de actualización del evento
     */
    @GetMapping("/event/edit/{id}")
    public String updateEvent(@PathVariable long id, Model model, Authentication authentication) {
    	Event event = eventServices.findById(id);

        // El usuario debe ser autenticado como owner del evento
        String username = authentication.getName();
        if (!event.getOwner().getUsername().equals(username)) {
            return "redirect:/login?error=Debes+ser+el+propietario+del+evento+para+editarlo";
        }

        // Crear DTO con los valores del evento
        NewEventDTO dto = new NewEventDTO();
        model.addAttribute("id", id);
        dto.setTitle(event.getTitle());
        dto.setDescription(event.getDescription());
        model.addAttribute("imagen", event.getImagen());
        dto.setStartDate(event.getStartDate());
        dto.setEndDate(event.getEndDate());
        dto.setMaxCapacity(event.getMaxCapacity());
        dto.setUbication(event.getUbication());

        // Convertir las categorías del evento a lista de Strings
        List<String> categories = event.getCategories().stream()
            .map(ec -> ec.getCategory().name())
            .collect(Collectors.toList());
        dto.setCategories(categories);

        model.addAttribute("eventDTO", dto);
        return "updateEvent";
    }
    
    /**
     * Actualiza la información del evento
     * 
     * @param id ID del evento que se quiere actualizar
     * @param updatedEvent Objeto que contiene los datos actualizados
     * @param authentication Sistema de autentificación de spring boot 
     * @return Página de información del evento, con informacion actualizada
     * @throws IOException En caso de problemas durante la gestión de la imagen
     */
	@PostMapping("/update-event/{id}")
	public String updateEvent(@PathVariable long id, 
			@ModelAttribute NewEventDTO updatedEvent, 
			Authentication authentication) throws IOException {
		
	    Event existing = eventServices.findById(id);
	    
	    // Comprueba que el dueño del evento es el que lo está editando
        User user = null;
        if (authentication != null && authentication.isAuthenticated()) {
            user = userServices.findByUsername(authentication.getName());
        }

        // Comprueba si es el dueño para mostrar datos de organizador del evento
        if (!existing.getOwner().equals(user) && user.getRol() != UserRol.ORGANIZADOR) {
            return "redirect:/login?error=Debes+ser+dueño+del+evento";
        }
	    
	    // Manejar la imagen
        MultipartFile nuevaImagen = updatedEvent.getImagen();
        
        if (nuevaImagen == null || nuevaImagen.isEmpty()) {
            existing.setImagen(existing.getImagen()); // Se mantiene la que había
        } else {
        	String nombreArchivo;
        	
        	// Imagen por defecto
        	if(updatedEvent.getImagen().isEmpty()) {
        		nombreArchivo = "default-event.png";
        	}
        	
        	nombreArchivo = UUID.randomUUID() + "_" + updatedEvent.getImagen().getOriginalFilename();
             
            // Ruta dentro del contenedor
            Path uploadPath = Paths.get(uploadDir + "/img/event/");
            Files.createDirectories(uploadPath);
            
            // Elimina la imagen anterior si no es la por defecto
            if (existing.getImagen() != null && !existing.getImagen().equals("default-event.png")) {
                Path oldImagePath = uploadPath.resolve(existing.getImagen());
                try {
                    Files.deleteIfExists(oldImagePath);
                } catch (IOException e) {
                    System.out.println("No se pudo eliminar la imagen anterior: " + e.getMessage());
                }
            }
             
            // Sube la imagen al servidor
            Path filePath = uploadPath.resolve(nombreArchivo);
            Files.copy(updatedEvent.getImagen().getInputStream(), filePath, StandardCopyOption.REPLACE_EXISTING);
        
            existing.setImagen(nombreArchivo);
        }
	    
	    // Datos actualizados
	    existing.setTitle(updatedEvent.getTitle());
	    existing.setDescription(updatedEvent.getDescription());
	    existing.setStartDate(updatedEvent.getStartDate());
	    existing.setEndDate(updatedEvent.getEndDate());
	    existing.setMaxCapacity(updatedEvent.getMaxCapacity());
	    
	    // Convertir Strings a Categories
        List<Categories> categoryEnums = new ArrayList<>();
        if(updatedEvent.getCategories() != null) {
            for(String catStr : updatedEvent.getCategories()) {
                try {
                    Categories catEnum = Categories.valueOf(catStr);
                    categoryEnums.add(catEnum);
                } catch (IllegalArgumentException e) {
                    System.out.println("Categoria no valida: " + catStr);
                }
            }
        }
        
        // Crear EventCategory para cada enum
        List<EventCategory> eventCategories = categoryEnums.stream()
            .map(catEnum -> new EventCategory(catEnum, existing))
            .collect(Collectors.toList());
        
        // Borra las que había antes y añade las nuevas
        existing.getCategories().clear(); 
        existing.getCategories().addAll(eventCategories);
        
        // Actualiza los datos del evento
	    eventServices.saveEvent(existing);
	    return "redirect:/event/info/" + id;
	}
}