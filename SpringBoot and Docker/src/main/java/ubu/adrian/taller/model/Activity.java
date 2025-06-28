package ubu.adrian.taller.model;

import jakarta.persistence.*;
import java.time.LocalTime;


/**
 * Entidad que modela una actividad
 */
@Entity
@Table(name = "activities")
public class Activity {
    // ID de la actividad
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private long id;

    // 
    @Column(nullable = false)
    private String name;

    @Column(length = 1000)
    private String description;

    @Column(name = "start_time")
    private LocalTime startTime;

    @Column(name = "end_time")
    private LocalTime endTime;

    @ManyToOne
    @JoinColumn(name = "event_id", nullable = false)
    private Event event;

    /**
     * Obtiene el ID
     * 
     * @return ID de la actividad
     */
    public long getId() {
        return id;
    }

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