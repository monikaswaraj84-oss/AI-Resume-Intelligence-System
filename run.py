"""Launcher for AI Resume ATS Analyzer.

Runs the FastAPI application and automatically opens your default web browser
to http://127.0.0.1:8000.
"""
import os
import sys
import threading
import time
import webbrowser
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def _open_browser():
    time.sleep(1.5)
    try:
        webbrowser.open("http://127.0.0.1:8000")
    except Exception:
        pass


if __name__ == "__main__":
    print("=" * 65)
    print("  AI RESUME ATS ANALYZER - SERVER RUNNING")
    print("=" * 65)
    print("  [*] Web App URL:  http://127.0.0.1:8000/")
    print("  [*] Alternative:  http://localhost:8000/")
    print("  NOTE: Port is 8000. Access via your local browser.")
    print("  Opening your default browser automatically...")
    print("  Press CTRL+C to terminate the server.")
    print("=" * 65)
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
