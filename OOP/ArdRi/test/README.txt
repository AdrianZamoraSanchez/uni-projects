Orden sugerido de implementación para la resolución de la práctica (y consiguiente ejecución de los tests correspondientes).

Level 1: Color, Coordenada, Sentido, TipoCelda y TipoPieza
Level 2: Traductor
Level 3: CoordenadasIncorrectasException || TipoArbitroException
Level 4: Pieza 
Level 5: Celda y Jugada
Level 6: Tablero
Level 7: Arbitro, ArbitroAbstracto, ArbitroArdRi, ArbitroBrandubh (tests básicos)
Level 8: Arbitro, ArbitroAbstracto, ArbitroArdRi, ArbitroBrandubh (tests básicos y medios)
Level 9: Arbitro, ArbitroAbstracto, ArbitroArdRi, ArbitroBrandubh (tests básicos, medios y avanzados)

Para ejecutar los tests de cada nivel ("Level") se proporcionan "suites" correspondientes (e.g. SuiteLevel1Test).

En el caso del Arbitro se descomponen los tests en distintos niveles, de menor a mayor dificultad.