package ubu.adrian.taller.dto;

import java.time.LocalTime;


import ubu.adrian.taller.model.Event;

public class NewActivityDTO {
	// Mismos atributos que Activity
    private String name;
    private String description;
    private LocalTime startTime;
    private LocalTime endTime;
    private Event event;

    /**
     * Devuelve el nombre de la actividad
     * 
     * @return Nombre de la actividad
     */
    public String getName() {
        return name;
    }

    /**
     * Establece el nombre
     * 
     * @param name Nombre que se establece
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Devuelve la descripción
     * 
     * @return Descripción de la actividad
     */
    public String getDescription() {
        return description;
    }

    /**
     * Establece la descripción
     * 
     * @param description Descripción que se establece
     */
    public void setDescription(String description) {
        this.description = description;
    }

    /**
     * Devuelve la hora de inicio
     * 
     * @return Hora de inicio
     */
    public LocalTime getStartTime() {
        return startTime;
    }

    /**
     * Establece la hora de inicio
     * 
     * @param startTime Hora de inicio a establecer
     */
    public void setStartTime(LocalTime startTime) {
        this.startTime = startTime;
    }

    /**
     * Devuelve la hora de fin
     * 
     * @return Hora de fin
     */
    public LocalTime getEndTime() {
        return endTime;
    }

    /**
     * Establece la hora de fin
     * 
     * @param endTime la hora de fin a establecer
     */
    public void setEndTime(LocalTime endTime) {
        this.endTime = endTime;
    }

    /**
     * Devuelve el evento
     * 
     * @return El evento
     */
    public Event getEvent() {
        return event;
    }

    /**
     * Establece el evento
     * 
     * @param Event Evento a establecer
     */
    public void setEvent(Event event) {
        this.event = event;
    }
}
