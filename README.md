
    python -m venv .ven

    . ./.venv/bin/activate

    pip install -r requirements.txt

    pip freeze >requirements.txt 

    docker compose up --remove-orphans streamlit