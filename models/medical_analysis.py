import numpy as np
import pandas as pd
import requests
import os

def detect_arrhythmias(ekg_signal):
    """
    Detects potential arrhythmias in the EKG signal using a simple threshold-based method.
    
    Parameters:
    ekg_signal (pd.Series): The EKG signal data as a pandas Series.
    
    Returns:
    dict: A dictionary containing the results of the analysis, including detected arrhythmias.
    """
    results = {
        'arrhythmias': [],
        'abnormal_beats': 0
    }
    
    # Simple threshold for abnormal beats (this is a placeholder; real analysis would be more complex)
    threshold = 1.5 * np.std(ekg_signal)
    
    for i in range(1, len(ekg_signal) - 1):
        if abs(ekg_signal[i] - ekg_signal[i - 1]) > threshold:
            results['arrhythmias'].append(i)
            results['abnormal_beats'] += 1
            
    return results

def analyze_ekg(ekg_signal):
    """
    Performs a comprehensive analysis of the EKG signal.
    
    Parameters:
    ekg_signal (pd.Series): The EKG signal data as a pandas Series.
    
    Returns:
    dict: A dictionary containing various analysis results.
    """
    analysis_results = {
        'max_value': ekg_signal.max(),
        'min_value': ekg_signal.min(),
        'mean_value': ekg_signal.mean(),
        'arrhythmia_detection': detect_arrhythmias(ekg_signal)
    }
    
    return analysis_results

def analyze_ekg_with_llm(ekg_signal, model="llama3", custom_prompt=None):
    """
    Sendet EKG-Daten an ein lokales LLM via Ollama zur medizinischen Analyse (Antwort auf Deutsch).
    """
    summary = f"Max: {ekg_signal.max():.2f}, Min: {ekg_signal.min():.2f}, Mittelwert: {ekg_signal.mean():.2f}"
    signal_str = ", ".join([f"{v:.2f}" for v in ekg_signal])

    if custom_prompt:
        prompt = custom_prompt.replace("{summary}", summary).replace("{signal}", signal_str).replace("{count}", str(len(ekg_signal)))
    else:
        prompt = (
            "Hier sind EKG-Daten einer Apple Watch als Mikrovolt-Zeitreihe.\n"
            f"Zusammenfassung: {summary}\n"
            f"Signal ({len(ekg_signal)} Werte): {signal_str}\n"
            "Bitte analysiere das Signal medizinisch und gib Hinweise auf mögliche Auffälligkeiten. "
            "Antworte ausschließlich auf Deutsch."
        )

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=600
    )
    response.raise_for_status()
    result = response.json()
    return result.get("response", "Keine Antwort vom Modell erhalten.")