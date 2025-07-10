import base64
import requests
import os

def analyze_image_with_llm(image_bytes, model="llava", custom_prompt=None):
    """
    Sendet Bilddaten an ein lokales LLM via Ollama zur Bildanalyse.
    
    Args:
        image_bytes: Bilddaten als Bytes
        model: Ollama-Modell für Bildanalyse
        custom_prompt: Benutzerdefinierter Prompt
    
    Returns:
        str: Analyseergebnis oder Fehlermeldung
    """
    # Bild zu Base64 konvertieren
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    if not custom_prompt:
        custom_prompt = "Analysiere dieses Bild medizinisch. Beschreibe was du siehst und gib Hinweise auf mögliche medizinische Befunde. Antworte auf Deutsch."
    
    try:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": custom_prompt,
                "images": [image_base64],
                "stream": False
            },
            timeout=600
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "Keine Antwort vom Modell erhalten.")
    except Exception as e:
        return f"Fehler bei Bildanalyse: {e}"