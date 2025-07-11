import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import io
import requests
import os
import json
from models.medical_analysis import analyze_ekg, analyze_ekg_with_llm
from models.image_analysis import analyze_image_with_llm
from utils.csv_parser import parse_apple_watch_csv
from utils.image_utils import fix_image_orientation

@st.cache_data
def load_i18n():
    with open('i18n.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def t(key, lang='de', **kwargs):
    texts = load_i18n()
    text = texts.get(lang, {}).get(key, key)
    return text.format(**kwargs) if kwargs else text

VISION_MODELS = {"llava", "bakllava", "llava-phi3", "llava-llama3", "medgemma", "medllava"}
HF_VISION_MODELS = [
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large",
    "microsoft/git-base",
    "microsoft/git-large"
]

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

def filter_vision_models(models):
    return [m for m in models if any(vm in m.lower() for vm in VISION_MODELS)]

def get_hf_vision_models():
    return HF_VISION_MODELS

# Sprachauswahl mit Browser-Speicherung
if 'language' not in st.session_state:
    st.session_state.language = st.query_params.get('lang', 'de')

selected_language = st.selectbox(
    t("language_selector", st.session_state.language),
    options=["de", "en", "es", "fr", "ru"],
    format_func=lambda x: {"de": "Deutsch", "en": "English", "es": "Español", "fr": "Français", "ru": "Русский"}[x],
    index=["de", "en", "es", "fr", "ru"].index(st.session_state.language)
)

if selected_language != st.session_state.language:
    st.query_params['lang'] = selected_language
    st.session_state.language = selected_language
    st.rerun()

language = st.session_state.language

st.title("Medical Analysis Platform")

# Hauptmenü
menu_option = st.selectbox(
    t("menu_select", language),
    options=["ekg_analysis", "image_analysis"],
    format_func=lambda x: {
        "ekg_analysis": t("menu_ekg", language),
        "image_analysis": t("menu_image", language)
    }[x]
)

if menu_option == "ekg_analysis":
    st.header(t("title", language))
    uploaded_file = st.file_uploader(t("file_uploader", language), type=["csv"])
    
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        metadata, ekg_signal, sampling_rate = parse_apple_watch_csv(content)
        # Metadaten anzeigen
        if metadata:
            st.subheader(t("metadata", language))
            for key, value in metadata.items():
                st.write(f"**{key}:** {value}")
        
        if ekg_signal.empty:
            st.error(t("no_data_error", language))
        else:
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

elif menu_option == "image_analysis":
    st.header(t("image_analysis_title", language))
    uploaded_image = st.file_uploader(t("image_uploader", language), type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Bildorientierung korrigieren
        corrected_image_bytes = fix_image_orientation(uploaded_image.read())
        uploaded_image.seek(0)  # Reset für spätere Verwendung
        
        st.image(corrected_image_bytes, caption=t("uploaded_image_caption", language), use_container_width=True)
        
        # Prompt-Editor für Bildanalyse
        custom_image_prompt = st.text_area(
            t("image_prompt_label", language),
            value=t("image_prompt_template", language),
            height=150,
            key=f"image_prompt_{language}"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            hf_models = get_hf_vision_models()
            selected_image_model = st.selectbox(t("available_vision_models", language), hf_models)
        
        with col2:
            if "image_llm_running" not in st.session_state:
                st.session_state.image_llm_running = False

            def start_image_llm():
                st.session_state.image_llm_running = True

            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
            image_llm_button = st.button(
                t("start_image_analysis", language),
                disabled=st.session_state.image_llm_running,
                on_click=start_image_llm
            )

        if st.session_state.image_llm_running:
            with st.spinner(t("image_analysis_running", language, model=selected_image_model)):
                image_bytes = uploaded_image.read()
                corrected_bytes = fix_image_orientation(image_bytes)
                llm_result = analyze_image_with_llm(corrected_bytes, model=selected_image_model, custom_prompt=custom_image_prompt)
                st.session_state.image_llm_running = False
                st.write(f"**{t('image_analysis_results', language)}**")
                st.write(llm_result)