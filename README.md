
    python -m venv .ven

    . ./.venv/bin/activate

    pip install -r requirements.txt

    pip freeze >requirements.txt 

    streamlit run app.py
    
    docker compose up --remove-orphans streamlit