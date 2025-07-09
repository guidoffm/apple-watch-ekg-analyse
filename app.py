import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import io
from models.medical_analysis import analyze_ekg, analyze_ekg_with_llm

st.title("Apple Watch EKG-Analyse (Originalformat)")

uploaded_file = st.file_uploader("Wähle eine Apple Watch EKG-Datei (CSV) zum Hochladen", type=["csv"])

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    sampling_rate = None
    lines = content.splitlines()
    data_lines = []
    for line in lines:
        line = line.strip()
        if line.lower().startswith("messrate"):
            try:
                parts = line.split(",")
                if len(parts) > 1:
                    sampling_rate = float(parts[1].split()[0].replace(",", "."))
            except Exception:
                pass
        elif line.lower().startswith("samplingrate") or line.lower().startswith("sampling_rate"):
            try:
                sampling_rate = float(line.split(":")[1].strip())
            except Exception:
                pass
        if line.replace('.', '', 1).replace(',', '', 1).replace('-', '', 1).isdigit():
            data_lines.append(line.replace(',', '.'))
    if not data_lines:
        st.error("Keine Messwerte gefunden. Bitte prüfe die Datei.")
    else:
        ekg_signal = pd.Series([float(val) for val in data_lines])
        st.write("Anzahl Messwerte:", len(ekg_signal))
        if sampling_rate:
            st.write(f"Messrate: {sampling_rate} Hz")
        else:
            st.warning("Messrate konnte nicht aus den Metadaten gelesen werden. X-Achse zeigt Samples.")

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
        if "llm_running" not in st.session_state:
            st.session_state.llm_running = False

        def start_llm():
            st.session_state.llm_running = True

        llm_button = st.button(
            "LLM-Analyse starten",
            disabled=st.session_state.llm_running,
            on_click=start_llm
        )

        if st.session_state.llm_running:
            with st.spinner("LLM-Analyse läuft..."):
                llm_result = analyze_ekg_with_llm(ekg_signal)
                st.write("LLM-Analyse:")
                st.write(llm_result)
            st.session_state.llm_running = False

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