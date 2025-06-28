package ubu.adrian.taller.dto;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;

import org.springframework.web.multipart.MultipartFile;

import jakarta.persistence.PostLoad;
import ubu.adrian.taller.model.Activity;
import ubu.adrian.taller.model.EventCategory;
import ubu.adrian.taller.model.User;

public class NewEventDTO {
	// Mismos atributos que Event
    private String title;
    private String description;
    private MultipartFile imagen;
    private LocalDate startDate;
    private LocalDate endDate;
    private String ubication;
    private List<String> categories = new ArrayList<>();
    private int maxCapacity;
    
    /**
     * Devuelve el título del evento
     * 
     * @return Título del evento
     */
    public String getTitle() {
        return title;
    }

    /**
     * Establece el título del evento
     * 
     * @param title Título del evento
     */
    public void setTitle(String title) {
        this.title = title;
    }

    /**
     * Devuelve la descripción del evento
     * 
     * @return Descripción del evento
     */
    public String getDescription() {
        return description;
    }

    /**
     * Establece la descripción del evento
     * 
     * @param description Descripción del evento
     */
    public void setDescription(String description) {
        this.description = description;
    }

    /**
     * Devuelve la fecha de inicio del evento
     * 
     * @return Fecha de inicio
     */
    public LocalDate getStartDate() {
        return startDate;
    }

    /**
     * Establece la fecha de inicio del evento
     * 
     * @param startDate Fecha de inicio
     */
    public void setStartDate(LocalDate startDate) {
        this.startDate = startDate;
    }

    /**
     * Devuelve la fecha de finalización del evento
     * 
     * @return Fecha de fin
     */
    public LocalDate getEndDate() {
        return endDate;
    }

    /**
     * Establece la fecha de finalización del evento
     * 
     * @param endDate Fecha de fin
     */
    public void setEndDate(LocalDate endDate) {
        this.endDate = endDate;
    }

    /**
     * Devuelve el lugar del evento
     * 
     * @return Lugar del evento
     */
    public String getUbication() {
        return ubication;
    }

    /**
     * Establece el lugar del evento
     * 
     * @param ubication Lugar del evento
     */
    public void setUbication(String ubication) {
        this.ubication = ubication;
    }
    
    /**
     * Devuelve la ruta a la imagen del evento
     * 
     * @return imagen Ruta de la imagen
     */
    public MultipartFile getImagen() {
        return imagen;
    }

    /**
     * Establece la imagen asociada al evento
     * 
     * @param imagen Ruta a la imagen del evento
     */
    public void setImagen(MultipartFile imagen) {
        this.imagen = imagen;
    }
    
    /**
     * Devuelve el aforo máximo del evento
     * 
     * @return Aforo del evento
     */
    public int getMaxCapacity() {
        return maxCapacity;
    }

    /**
     * Establece el aforo máximo del evento
     * 
     * @param maxCapacity Aforo del evento
     */
    public void setMaxCapacity(int maxCapacity) {
        this.maxCapacity = maxCapacity;
    }

   
    /**
     * Establece las categorías del evento
     * 
     * @param categories Categorías del evento
     */
	public void setCategories(List<String> categories) {
		this.categories = categories;
		
	}
	
	/**
     * Devuelve las categorías del evento
     * 
     * @return categorías a las que pertenece el evento
     */
	public List<String> getCategories() {
		return categories;
	}
}
