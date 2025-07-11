from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import torch

# Global variables for model caching
_processor = None
_model = None

def load_model(model_name="Salesforce/blip-image-captioning-base"):
    """
    Lädt das Hugging Face Vision-Modell.
    """
    global _processor, _model
    if _processor is None or _model is None:
        _processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        _model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        # GPU verwenden falls verfügbar
        if torch.cuda.is_available():
            _model = _model.cuda()
    
    return _processor, _model

def analyze_image_with_llm(image_bytes, model="Salesforce/blip-image-captioning-base", custom_prompt=None):
    """
    Analysiert Bilddaten mit einem Hugging Face Vision-Modell.
    
    Args:
        image_bytes: Bilddaten als Bytes
        model: Hugging Face Modell-Name
        custom_prompt: Benutzerdefinierter Prompt (wird als Conditional Text verwendet)
    
    Returns:
        str: Analyseergebnis oder Fehlermeldung
    """
    try:
        # Bild laden
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Modell laden
        processor, model_instance = load_model(model)
        
        # Bild verarbeiten (ohne Text für unconditional generation)
        inputs = processor(image, return_tensors="pt")
        
        # GPU verwenden falls verfügbar
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generierung
        with torch.no_grad():
            out = model_instance.generate(**inputs, max_length=150, num_beams=5, early_stopping=True)
        
        # Ergebnis dekodieren
        result = processor.decode(out[0], skip_special_tokens=True)
        
        return result
        
    except Exception as e:
        return f"Fehler bei Bildanalyse: {e}"