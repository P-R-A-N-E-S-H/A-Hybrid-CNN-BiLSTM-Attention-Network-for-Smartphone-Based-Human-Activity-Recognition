"""
======================================================================
run_demo.py - One-Click Final Review Live Demo Launcher
======================================================================

Starts the real-time FastAPI / WebSocket server and automatically opens
the interactive glassmorphic review dashboard in the default browser.

Author: Pranesh
======================================================================
"""

import os
import sys
import time
import webbrowser
import uvicorn
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config


def launch():
    print("=" * 70)
    print(" HAR Deep Learning & IoT Telemetry System - Final Review Launcher")
    print("=" * 70)
    print(f"Server URL: http://localhost:{config.API_PORT}")
    print("Opening web browser dashboard...")

    def open_browser():
        time.sleep(1.2)
        webbrowser.open(f"http://localhost:{config.API_PORT}")

    import threading
    from api.server import app
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)


if __name__ == "__main__":
    launch()
