# Architecture

Streamlit UI → FastAPI service (main.py) → loads model.pt + scaler.pkl + meta.pkl
from models/ at startup → returns prediction. Packaged with a Dockerfile.
No database, no auth — out of scope for this test.
