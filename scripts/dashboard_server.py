import os
import json
import glob
from pathlib import Path
from flask import Flask, send_from_directory, jsonify
from threading import Thread

ROOT_DIR = Path(__file__).resolve().parent.parent

app = Flask(__name__, static_folder=str(ROOT_DIR / "dashboard"), static_url_path="/")
PORT = 8080

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route("/api/state")
def get_state():
    """Finds the most recent bot_state.json and returns it."""
    state_files = glob.glob(str(ROOT_DIR / "bot_state_*.json"))
    if not state_files:
        default_state = ROOT_DIR / "bot_state.json"
        if default_state.exists():
            state_files = [str(default_state)]
        else:
            return jsonify({"error": "No bot state found"}), 404
            
    # Sort by modification time to get the latest
    latest_file = max(state_files, key=os.path.getmtime)
    
    try:
        with open(latest_file, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_flask():
    # Disable werkzeug logging to keep the console clean
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # Silence the "* Serving Flask app..." startup banner
    from flask import cli
    cli.show_server_banner = lambda *args: None
    
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

def start_dashboard_server():
    print(f"\n🌐 Starting Live Dashboard Server at http://localhost:{PORT}")
    thread = Thread(target=run_flask, daemon=True)
    thread.start()
    
    public_url = None
    try:
        from pycloudflared import try_cloudflare
        tunnel = try_cloudflare(port=PORT)
        public_url = tunnel.tunnel
        print(f"🌍 Public Dashboard Tunnel Created: {public_url}")
    except ImportError:
        print("💡 Tip: 'pip install pycloudflared' to get a public dashboard link.")
    except Exception as e:
        print(f"⚠️ Could not start Cloudflare tunnel: {e}")
        
    return thread, public_url
