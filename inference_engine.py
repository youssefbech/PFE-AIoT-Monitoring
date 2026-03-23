import pandas as pd
import paho.mqtt.client as mqtt
import json
import time
import joblib
import numpy as np

# --- CONFIGURATION ---
MQTT_BROKER = "localhost"
MQTT_TOPIC = "factory/machine_1/inference"
MODEL_PATH = "svm_model.pkl" 
SCALER_PATH = "scaler.pkl"  # Le fichier généré lors de votre standardisation

# 1. Chargement du modèle et du scaler (la "recette" de normalisation)
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Modèle et Scaler chargés avec succès.")
except FileNotFoundError:
    print("❌ Erreur : Assurez-vous d'avoir 'svm_model.pkl' et 'scaler.pkl' dans le dossier.")
    exit()

# 2. Configuration MQTT
client = mqtt.Client()
try:
    client.connect(MQTT_BROKER, 1883, 60)
except:
    print("❌ Erreur : Impossible de se connecter au broker MQTT (Mosquitto lancé ?)")
    exit()

# 3. Chargement des données RÉELLES (non transformées)
# Utilisez le CSV d'origine pour que le dashboard affiche des valeurs compréhensibles
df = pd.read_csv('bldc_predictive_maintenance_dataset (1) (1) (1).csv')

# Liste des colonnes d'entrée (Features) - Doit être identique à l'entraînement
features_cols = ['Timestamp', 'Current_A', 'Voltage_V', 'Temperature_C', 'Power_W', 'Delta_V']

print("🚀 Lancement du monitoring avec conversion en temps réel...")

try:
    for index, row in df.iterrows():
        # --- ÉTAPE A : Extraction des valeurs réelles ---
        # On crée un tableau avec les valeurs brutes du CSV
        raw_features = row[features_cols].values.reshape(1, -1)
        
        # --- ÉTAPE B : Standardisation pour le modèle ---
        # On applique la transformation juste pour l'IA
        scaled_features = scaler.transform(raw_features)
        
        # --- ÉTAPE C : Inférence ---
        prediction = model.predict(scaled_features)[0]
        
        # --- ÉTAPE D : Préparation du message (Valeurs RÉELLES pour l'humain) ---
        # On utilise row['Voltage_V'] et non scaled_features pour l'affichage
        payload = {
            "timestamp_csv": row['Timestamp'],
            "current": float(row['Current_A']),    # ex: 2.5 au lieu de -0.12
            "voltage": float(row['Voltage_V']),    # ex: 230.0 au lieu de 0.45
            "temp": float(row['Temperature_C']),   # ex: 45.0 au lieu de 1.2
            "power": float(row['Power_W']),        #
            "anomaly_score": int(prediction),      # 0 ou 1
            "real_time": time.strftime("%H:%M:%S")
        }
        
        # Envoi au Dashboard via MQTT
        client.publish(MQTT_TOPIC, json.dumps(payload))
        
        # Affichage console pour suivi
        status = "✅ NOMINAL" if prediction == 0 else "⚠️ DÉFAUT"
        print(f"[{payload['real_time']}] V: {payload['voltage']}V | T: {payload['temp']}°C | Statut: {status}")
        
        time.sleep(0.1) # Simule une lecture chaque seconde

except KeyboardInterrupt:
    print("\n🛑 Arrêt du moteur d'inférence.")
    client.disconnect()
except Exception as e:
    print(f"❌ Erreur durant l'exécution : {e}")