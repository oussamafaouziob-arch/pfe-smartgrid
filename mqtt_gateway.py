import json
import sqlite3
import os
import threading
import time
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import paho.mqtt.client as mqtt
from flask import Flask, request, jsonify

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_SUB = "smartgrid/+/data"
MQTT_TOPIC_PUB = "smartgrid/alerts"

MODEL_PATH = Path(__file__).parent / "random_forest_smartgrid.joblib"
DB_PATH = Path(__file__).parent / "smartgrid.db"

TEMP_CRITICAL_C = 80.0
CURRENT_CRITICAL_A = 20.0


def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT,
            timestamp TEXT,
            current_A REAL,
            voltage_V REAL,
            temperature_C REAL,
            prediction TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()


def save_measurement(node_id, current, voltage, temperature, prediction, confidence):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """INSERT INTO measurements
           (node_id, timestamp, current_A, voltage_V, temperature_C, prediction, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            node_id,
            datetime.now(timezone.utc).isoformat(),
            current,
            voltage,
            temperature,
            prediction,
            confidence,
        ),
    )
    conn.commit()
    conn.close()


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modele introuvable: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def build_features(current, voltage, temperature, model_columns):
    features = {}
    for col in model_columns:
        if col.startswith("tau"):
            features[col] = 5.0
        elif col.startswith("p"):
            features[col] = voltage / 10.0
        elif col.startswith("g"):
            features[col] = current / 10.0
        else:
            features[col] = 0.0
    return np.array([[features[c] for c in model_columns]])


model = None
model_columns = None


def on_connect(client, userdata, flags, rc):
    print(f"Connecte au broker MQTT (rc={rc}), abonnement a {MQTT_TOPIC_SUB}")
    client.subscribe(MQTT_TOPIC_SUB)


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        print(f"Message invalide recu sur {msg.topic}: {msg.payload}")
        return

    node_id = payload.get("node_id", "unknown")
    current = float(payload.get("current_A", 0))
    voltage = float(payload.get("voltage_V", 0))
    temperature = float(payload.get("temperature_C", 0))

    if temperature >= TEMP_CRITICAL_C or current >= CURRENT_CRITICAL_A:
        prediction = "PANNE"
        confidence = 1.0
    else:
        X = build_features(current, voltage, temperature, model_columns)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        prediction = pred
        confidence = float(np.max(proba))

    print(f"[{node_id}] I={current:.2f}A V={voltage:.1f}V T={temperature:.1f}C -> {prediction} ({confidence:.2f})")

    save_measurement(node_id, current, voltage, temperature, prediction, confidence)

    alert_payload = json.dumps({
        "node_id": node_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current_A": current,
        "voltage_V": voltage,
        "temperature_C": temperature,
        "prediction": prediction,
        "confidence": confidence,
    })
    client.publish(MQTT_TOPIC_PUB, alert_payload)


# ---------------- API HTTP (pour le dashboard / historique) ----------------

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    # Autorise le dashboard (heberge ailleurs, ex. GitHub Pages) a appeler cette API
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    return response


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/history")
def history():
    """
    Parametres (query string) :
      start = date de debut, format YYYY-MM-DD (obligatoire)
      end   = date de fin, format YYYY-MM-DD (optionnel, sinon = start)
      node_id = filtrer par noeud (optionnel)
    Exemple : /api/history?start=2026-06-05&end=2026-06-12
    """
    start = request.args.get("start")
    end = request.args.get("end") or start
    node_id = request.args.get("node_id")

    if not start:
        return jsonify({"error": "Le parametre 'start' (YYYY-MM-DD) est requis"}), 400

    start_iso = f"{start}T00:00:00"
    end_iso = f"{end}T23:59:59"

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    query = """
        SELECT node_id, timestamp, current_A, voltage_V, temperature_C, prediction, confidence
        FROM measurements
        WHERE timestamp >= ? AND timestamp <= ?
    """
    params = [start_iso, end_iso]
    if node_id:
        query += " AND node_id = ?"
        params.append(node_id)
    query += " ORDER BY timestamp ASC"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    results = [dict(r) for r in rows]
    summary = {"NORMAL": 0, "ANOMALIE": 0, "PANNE": 0}
    for r in results:
        if r["prediction"] in summary:
            summary[r["prediction"]] += 1

    return jsonify({"count": len(results), "summary": summary, "results": results})


def run_flask():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


def run_mqtt():
    global model, model_columns

    init_db()
    print("Base de donnees SQLite prete.")

    model = load_model()
    model_columns = list(model.feature_names_in_)
    print(f"Modele charge. Colonnes attendues: {model_columns}")

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    print("Gateway demarree. En attente de messages...")
    client.loop_forever()


def main():
    # Lance le serveur API (Flask) dans un thread separe,
    # et la boucle MQTT dans le thread principal
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    time.sleep(1)
    run_mqtt()


if __name__ == "__main__":
    main()
