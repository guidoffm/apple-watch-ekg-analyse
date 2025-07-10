import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import io
import requests
import os
from models.medical_analysis import analyze_ekg, analyze_ekg_with_llm

@st.cache_data(ttl=60)
def get_ollama_models():
    try:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        return sorted([model["name"] for model in models])
    except:
        return ["llama3"]

st.title("Apple Watch EKG-Analyse (Originalformat)")

uploaded_file = st.file_uploader("Wähle eine Apple Watch EKG-Datei (CSV) zum Hochladen", type=["csv"])

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
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
                        sampling_rate = float(value.split()[0])
                    except:
                        pass
        elif line.replace('.', '', 1).replace(',', '', 1).replace('-', '', 1).isdigit():
            data_lines.append(line.replace(',', '.'))
    # Metadaten anzeigen
    if metadata:
        st.subheader("Metadaten")
        for key, value in metadata.items():
            st.write(f"**{key}:** {value}")
    
    if not data_lines:
        st.error("Keine Messwerte gefunden. Bitte prüfe die Datei.")
    else:
        ekg_signal = pd.Series([float(val) for val in data_lines])
        st.subheader("Signalanalyse")
        st.write("Anzahl Messwerte:", len(ekg_signal))
        if sampling_rate:
            st.write(f"Messrate: {sampling_rate} Hz")
        else:
            st.warning("Messrate konnte nicht aus den Metadaten gelesen werden. X-Achse zeigt Samples.")

        st.write("**Einfache Analyse:**")
        st.write(f"Maximalwert: {ekg_signal.max():.2f} µV")
        st.write(f"Minimalwert: {ekg_signal.min():.2f} µV")
        st.write(f"Mittelwert: {ekg_signal.mean():.2f} µV")

        # Medizinische Analyse
        analysis_results = analyze_ekg(ekg_signal)
        st.write("**Medizinische Analyse Ergebnisse:**")
        for key, value in analysis_results.items():
            st.write(f"{key}: {value}")

        # LLM-Analyse
        st.subheader("LLM-Analyse")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            available_models = get_ollama_models()
            selected_model = st.selectbox("Verfügbare Ollama-Modelle:", available_models)
        
        with col2:
            if "llm_running" not in st.session_state:
                st.session_state.llm_running = False

            def start_llm():
                st.session_state.llm_running = True

            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
            llm_button = st.button(
                "LLM-Analyse starten",
                disabled=st.session_state.llm_running,
                on_click=start_llm
            )

        if st.session_state.llm_running:
            with st.spinner(f"LLM-Analyse läuft ({selected_model})..."):
                llm_result = analyze_ekg_with_llm(ekg_signal, model=selected_model)
                st.session_state.llm_running = False
                st.write("**LLM-Analyse:**")
                st.write(llm_result)

        # Interaktiver Plot mit Plotly
        st.subheader("Interaktiver EKG-Plot (mit Maus zoomen, Doppelklick oder Button zum Zurücksetzen)")
        if sampling_rate:
            import numpy as np
            zeit_achse = np.arange(len(ekg_signal)) / sampling_rate
            x = zeit_achse
            x_label = "Zeit (s)"
        else:
            x = list(range(len(ekg_signal)))
            x_label = "Zeit (Samples)"

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=ekg_signal, mode='lines', name='EKG'))
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title="Spannung (µV)",
            title="EKG-Signal (Apple Watch)",
            dragmode='zoom',  # Standardmäßig Zoom
            hovermode='x unified'
        )
        fig.update_xaxes(rangeslider_visible=True)

        st.plotly_chart(fig, use_container_width=True)