import streamlit as st
from transformers import pipeline

st.title("Datei-Analyse mit freiem LLM")

uploaded_file = st.file_uploader("Wähle eine Datei zum Hochladen", type=["txt", "md", "csv"])

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    st.write("Datei-Inhalt:")
    st.text_area("Inhalt", content, height=200)

    # Beispiel: Textzusammenfassung mit freiem LLM
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    if st.button("Analysiere & fasse zusammen"):
        summary = summarizer(content, max_length=130, min_length=30, do_sample=False)
        st.subheader("Zusammenfassung:")
        st.write(summary[0]['summary_text'])