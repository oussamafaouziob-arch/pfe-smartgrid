import json
import sqlite3
import os
import threading
import time
from collections import deque, defaultdict
from pathlib import Path
from datetime import datetime, timedelta, timezone

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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alarms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT,
            timestamp TEXT,
            type TEXT,
            message TEXT
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

last_alarm_type = {}


def alarm_type(prediction, warning_level):
    if prediction == "PANNE":
        return "PANNE"
    if warning_level == "ALERTE":
        return "ALERTE"
    if warning_level == "ATTENTION":
        return "ATTENTION"
    if prediction == "ANOMALIE":
        return "ANOMALIE"
    return "NORMAL"


def log_alarm_if_changed(node_id, prediction, trend, current, voltage, temperature):
    """Enregistre une alarme seulement quand l'etat du noeud change."""
    atype = alarm_type(prediction, trend["warning_level"])
    prev = last_alarm_type.get(node_id)
    last_alarm_type[node_id] = atype
    if prev == atype or (prev is None and atype == "NORMAL"):
        return
    msg = f"T={temperature:.1f} °C · I={current:.2f} A · V={voltage:.0f} V"
    if atype in ("ALERTE", "ATTENTION") and trend["eta_min"] is not None:
        msg += f" · seuil ({trend['eta_cause']}) dans {trend['eta_min']} min"
    if atype == "NORMAL":
        msg = "Retour à la normale · " + msg
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        "INSERT INTO alarms (node_id, timestamp, type, message) VALUES (?, ?, ?, ?)",
        (node_id, datetime.now(timezone.utc).isoformat(), atype, msg),
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


# ---------------- ALERTE PRECOCE PAR TENDANCE ----------------
# Idee : au lieu d'attendre que le seuil soit depasse, on estime "dans combien
# de minutes" il sera atteint si la tendance actuelle continue (extrapolation
# lineaire sur les derniers points). Ce n'est PAS un modele de ML de prevision.

WINDOW_SIZE = 60          # nombre de dernieres mesures gardees par noeud (12 x 5 s = 1 min)
MIN_POINTS = 15            # nombre minimal de mesures avant de calculer une tendance
HORIZON_MAX_MIN = 120     # au-dela de 2 h, on considere qu'il n'y a pas de risque proche
WARN_ATTENTION_MIN = 30   # niveau ATTENTION si seuil estime dans 30 min ou moins
WARN_ALERTE_MIN = 15      # niveau ALERTE si seuil estime dans 15 min ou moins
MIN_SLOPE_TEMP = 0.05     # pente minimale (degC/min) pour parler de "tendance a la hausse"
MIN_SLOPE_CURRENT = 0.02  # pente minimale (A/min)

# historique recent par noeud : liste de (temps_en_secondes, temperature, courant)
recent_points = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))


def fit_trend(points, index):
    """Regression lineaire sur la fenetre. Retourne (pente par minute, valeur lissee actuelle)."""
    t0 = points[0][0]
    xs = np.array([(p[0] - t0) / 60.0 for p in points])
    ys = np.array([p[index] for p in points])
    if np.ptp(xs) < 1e-6:
        return None, None
    slope, intercept = np.polyfit(xs, ys, 1)
    smoothed_now = slope * xs[-1] + intercept
    return float(slope), float(smoothed_now)


def minutes_to_threshold(slope, value, threshold, min_slope):
    """Temps estime (minutes) avant d'atteindre le seuil, ou None si pas de risque."""
    if slope is None or slope < min_slope:
        return None
    if value >= threshold:
        return None  # deja au-dessus : c'est l'etat PANNE qui s'en charge
    eta = (threshold - value) / slope
    if eta > HORIZON_MAX_MIN:
        return None
    return eta


def analyze_trend(node_id, temperature, current):
    """Met a jour l'historique du noeud et retourne l'etat de l'alerte precoce."""
    points = recent_points[node_id]
    points.append((time.time(), temperature, current))

    result = {
        "warning_level": "NONE",
        "eta_min": None,
        "eta_cause": None,
        "trend_temp_per_min": None,
        "trend_current_per_min": None,
    }
    if len(points) < MIN_POINTS:
        return result

    pts = list(points)
    slope_t, smooth_t = fit_trend(pts, 1)
    slope_i, smooth_i = fit_trend(pts, 2)
    result["trend_temp_per_min"] = None if slope_t is None else round(slope_t, 2)
    result["trend_current_per_min"] = None if slope_i is None else round(slope_i, 3)

    eta_t = minutes_to_threshold(slope_t, smooth_t, TEMP_CRITICAL_C, MIN_SLOPE_TEMP)
    eta_i = minutes_to_threshold(slope_i, smooth_i, CURRENT_CRITICAL_A, MIN_SLOPE_CURRENT)

    candidates = []
    if eta_t is not None:
        candidates.append((eta_t, "temperature"))
    if eta_i is not None:
        candidates.append((eta_i, "courant"))
    if not candidates:
        return result

    eta, cause = min(candidates)
    result["eta_min"] = round(eta, 1)
    result["eta_cause"] = cause
    if eta <= WARN_ALERTE_MIN:
        result["warning_level"] = "ALERTE"
    elif eta <= WARN_ATTENTION_MIN:
        result["warning_level"] = "ATTENTION"
    return result


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

    trend = analyze_trend(node_id, temperature, current)

    print(f"[{node_id}] I={current:.2f}A V={voltage:.1f}V T={temperature:.1f}C -> {prediction} ({confidence:.2f})"
          f" | alerte precoce: {trend['warning_level']}"
          + (f" (seuil dans {trend['eta_min']} min, cause: {trend['eta_cause']})" if trend["eta_min"] is not None else ""))

    save_measurement(node_id, current, voltage, temperature, prediction, confidence)
        log_alarm_if_changed(node_id, prediction, trend, current, voltage, temperature)

    alert_payload = json.dumps({
        "node_id": node_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current_A": current,
        "voltage_V": voltage,
        "temperature_C": temperature,
        "prediction": prediction,
        "confidence": confidence,
        "warning_level": trend["warning_level"],
        "eta_min": trend["eta_min"],
        "eta_cause": trend["eta_cause"],
        "trend_temp_per_min": trend["trend_temp_per_min"],
        "trend_current_per_min": trend["trend_current_per_min"],
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

@app.route("/api/recent")
def recent():
    """Dernieres mesures (pour recharger la vue en direct). Exemple : /api/recent?minutes=10"""
    try:
        minutes = min(int(request.args.get("minutes", 10)), 60)
    except ValueError:
        minutes = 10
    since = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT node_id, timestamp, current_A, voltage_V, temperature_C, prediction, confidence
           FROM measurements WHERE timestamp >= ? ORDER BY timestamp ASC""",
        (since,),
    ).fetchall()
    conn.close()

    return jsonify({"count": len(rows), "results": [dict(r) for r in rows]})
@app.route("/api/alarms")
def list_alarms():
    """Journal d'alarmes (changements d'etat). Exemple : /api/alarms?limit=200"""
    try:
        limit = min(int(request.args.get("limit", 200)), 1000)
    except ValueError:
        limit = 200
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT node_id, timestamp, type, message FROM alarms ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return jsonify({"count": len(rows), "results": [dict(r) for r in rows]})def run_flask():
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
