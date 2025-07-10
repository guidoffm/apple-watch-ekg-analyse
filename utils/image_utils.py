from PIL import Image
import io

def fix_image_orientation(image_bytes):
    """
    Korrigiert die Bildorientierung basierend auf EXIF-Daten.
    
    Args:
        image_bytes: Bilddaten als Bytes
        
    Returns:
        bytes: Korrigierte Bilddaten als Bytes
    """
    try:
        # Bild aus Bytes laden
        image = Image.open(io.BytesIO(image_bytes))
        
        # EXIF-Daten lesen und Orientierung korrigieren
        if hasattr(image, '_getexif') and image._getexif() is not None:
            exif = image._getexif()
            orientation = exif.get(274)  # 274 ist der EXIF-Tag für Orientierung
            
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
        
        # Korrigiertes Bild zurück zu Bytes konvertieren
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=95)
        return output.getvalue()
        
    except Exception:
        # Bei Fehlern Original-Bytes zurückgeben
        return image_bytes