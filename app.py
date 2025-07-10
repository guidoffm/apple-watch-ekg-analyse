import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import io
import requests
import os
import json
from models.medical_analysis import analyze_ekg, analyze_ekg_with_llm

@st.cache_data
def load_i18n():
    with open('i18n.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def t(key, lang='de', **kwargs):
    texts = load_i18n()
    text = texts.get(lang, {}).get(key, key)
    return text.format(**kwargs) if kwargs else text

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

# Sprachauswahl mit Browser-Speicherung
if 'language' not in st.session_state:
    st.session_state.language = st.query_params.get('lang', 'de')

language = st.selectbox(
    "Language / Sprache:",
    options=["de", "en", "es", "fr", "ru"],
    format_func=lambda x: {"de": "Deutsch", "en": "English", "es": "Español", "fr": "Français", "ru": "Русский"}[x],
    index=["de", "en", "es", "fr", "ru"].index(st.session_state.language)
)

if language != st.session_state.language:
    st.query_params['lang'] = language
    st.session_state.language = language

st.title(t("title", language))

uploaded_file = st.file_uploader(t("file_uploader", language), type=["csv"])

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
                        rate_str = value.split()[0].replace(',', '.')
                        sampling_rate = float(rate_str)
                    except:
                        pass
        elif line.replace('.', '', 1).replace(',', '', 1).replace('-', '', 1).isdigit():
            data_lines.append(line.replace(',', '.'))
    # Metadaten anzeigen
    if metadata:
        st.subheader(t("metadata", language))
        for key, value in metadata.items():
            st.write(f"**{key}:** {value}")
    
    if not data_lines:
        st.error(t("no_data_error", language))
    else:
        ekg_signal = pd.Series([float(val) for val in data_lines])
        st.subheader(t("signal_analysis", language))
        st.write(t("measurement_count", language), len(ekg_signal))
        if sampling_rate:
            st.write(t("sampling_rate", language, rate=sampling_rate))
        else:
            st.warning(t("sampling_rate_warning", language))

        st.write(f"**{t('simple_analysis', language)}**")
        st.write(t("max_value", language, value=ekg_signal.max()))
        st.write(t("min_value", language, value=ekg_signal.min()))
        st.write(t("mean_value", language, value=ekg_signal.mean()))

        # Medizinische Analyse
        analysis_results = analyze_ekg(ekg_signal)
        st.write(f"**{t('medical_analysis', language)}**")
        for key, value in analysis_results.items():
            st.write(f"{key}: {value}")

        # LLM-Analyse
        st.subheader(t("llm_analysis", language))
        
        custom_prompt = st.text_area(
            t("prompt_label", language),
            value=t("prompt_template", language),
            height=200,
            key=f"prompt_{language}"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            available_models = get_ollama_models()
            selected_model = st.selectbox(t("available_models", language), available_models)
        
        with col2:
            if "llm_running" not in st.session_state:
                st.session_state.llm_running = False

            def start_llm():
                st.session_state.llm_running = True

            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
            llm_button = st.button(
                t("start_analysis", language),
                disabled=st.session_state.llm_running,
                on_click=start_llm
            )

        if st.session_state.llm_running:
            with st.spinner(t("analysis_running", language, model=selected_model)):
                llm_result = analyze_ekg_with_llm(ekg_signal, model=selected_model, custom_prompt=custom_prompt)
                st.session_state.llm_running = False
                st.write(f"**{t('analysis_results', language)}**")
                st.write(llm_result)

        # Interaktiver Plot mit Plotly
        st.subheader(t("plot_title", language))
        if sampling_rate:
            import numpy as np
            zeit_achse = np.arange(len(ekg_signal)) / sampling_rate
            x = zeit_achse
            x_label = t("time_seconds", language)
        else:
            x = list(range(len(ekg_signal)))
            x_label = t("time_samples", language)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=ekg_signal, mode='lines', name='EKG'))
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=t("voltage", language),
            title=t("ekg_signal_title", language),
            dragmode='zoom',  # Standardmäßig Zoom
            hovermode='x unified'
        )
        fig.update_xaxes(rangeslider_visible=True)

        st.plotly_chart(fig, use_container_width=True)