package juego.modelo;

public enum Color {
	NEGRO('X'), BLANCO('O');
	
	private char letra;
	
	private Color(char letra) {
		this.letra = letra;
	}
	
	public char toChar() {
		return letra;
	}
}
	
	
