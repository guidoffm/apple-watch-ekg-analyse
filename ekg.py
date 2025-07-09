import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from models.medical_analysis import analyze_ekg, analyze_ekg_with_llm

st.title("Apple Watch EKG-Analyse (Originalformat)")

uploaded_file = st.file_uploader("Wähle eine Apple Watch EKG-Datei (CSV) zum Hochladen", type=["csv"])

if uploaded_file is not None:
    # Datei als Text einlesen
    content = uploaded_file.read().decode("utf-8")
    # Nur Zeilen mit Zahlen extrahieren (Messwerte)
    lines = content.splitlines()
    data_lines = []
    for line in lines:
        line = line.strip()
        # Zeile ist eine Zahl (ggf. mit Komma als Dezimaltrennzeichen)
        if line.replace('.', '', 1).replace(',', '', 1).replace('-', '', 1).isdigit():
            # Komma durch Punkt ersetzen (für float)
            data_lines.append(line.replace(',', '.'))
    if not data_lines:
        st.error("Keine Messwerte gefunden. Bitte prüfe die Datei.")
    else:
        # In DataFrame umwandeln
        ekg_signal = pd.Series([float(val) for val in data_lines])
        st.write("Anzahl Messwerte:", len(ekg_signal))

        st.write("Einfache Analyse:")
        st.write(f"Maximalwert: {ekg_signal.max():.2f} µV")
        st.write(f"Minimalwert: {ekg_signal.min():.2f} µV")
        st.write(f"Mittelwert: {ekg_signal.mean():.2f} µV")

        # Medizinische Analyse
        analysis_results = analyze_ekg(ekg_signal)
        st.write("Medizinische Analyse Ergebnisse:")
        for key, value in analysis_results.items():
            st.write(f"{key}: {value}")

        # LLM-Analyse
        api_key = st.text_input("OpenAI API Key", type="password")
        if st.button("LLM-Analyse starten") and api_key:
            llm_result = analyze_ekg_with_llm(ekg_signal, api_key)
            st.write("LLM-Analyse:")
            st.write(llm_result)

        # Plot
        fig, ax = plt.subplots()
        ax.plot(ekg_signal)
        ax.set_title("EKG-Signal (Apple Watch)")
        ax.set_xlabel("Zeit (Samples)")
        ax.set_ylabel("Spannung (µV)")
        st.pyplot(fig)