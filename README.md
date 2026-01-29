
    python -m venv .venv

    . ./.venv/bin/activate

    pip install -r requirements.txt

    streamlit run app.py
    
    docker compose up --remove-orphans streamlit

## Konfiguration

### Debug-Modus
Für detaillierte Debug-Ausgaben setzen Sie die Umgebungsvariable:

    export DEBUG=true
    streamlit run app.py

### Hugging Face Token
Für MedGemma wird ein Hugging Face Token benötigt:
1. Erstellen Sie einen Token auf https://huggingface.co/settings/tokens
2. Geben Sie den Token in der App ein (wird im Browser gespeichert)
3. Oder setzen Sie die Umgebungsvariable: `export HF_TOKEN=your_token_here`