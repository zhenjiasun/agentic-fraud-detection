"""Entry point: the Dash dashboard (expects the API running on :8000)."""
from src.dashboard.app import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
