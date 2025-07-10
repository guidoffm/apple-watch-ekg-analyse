import pandas as pd

def parse_apple_watch_csv(content):
    """
    Parst Apple Watch EKG CSV-Daten und extrahiert Metadaten und Signaldaten.
    
    Args:
        content (str): CSV-Dateiinhalt als String
        
    Returns:
        tuple: (metadata dict, ekg_signal pd.Series, sampling_rate float or None)
    """
    sampling_rate = None
    metadata = {}
    lines = content.splitlines()
    data_lines = []
    
    for line in lines:
        line = line.strip()
        if "," in line and not line.replace('.', '', 1).replace(',', '', 1).replace('-', '', 1).isdigit():
            parts = line.split(",", 1)
            if len(parts) == 2:
                key, value = parts[0].strip(), parts[1].strip().strip('"')
                metadata[key] = value
                if key.lower() == "messrate":
                    try:
                        rate_str = value.split()[0].replace(',', '.')
                        sampling_rate = float(rate_str)
                    except:
                        pass
        elif line.replace('.', '', 1).replace(',', '', 1).replace('-', '', 1).isdigit():
            data_lines.append(line.replace(',', '.'))
    
    ekg_signal = pd.Series([float(val) for val in data_lines]) if data_lines else pd.Series([])
    
    return metadata, ekg_signal, sampling_rate