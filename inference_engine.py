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

# 1. Chargement du modèle
model = joblib.load(MODEL_PATH)

# 2. MQTT Setup
client = mqtt.Client()
client.connect(MQTT_BROKER, 1883, 60)

# 3. Chargement des données
df = pd.read_csv('dataset_standardized.csv')

# Liste des colonnes d'entrée (Features) pour le SVC
# On exclut 'Fault_Label'
features_cols = ['Timestamp', 'Current_A', 'Voltage_V', 'Temperature_C', 'Power_W', 'Delta_V']

print("🚀 Lancement du monitoring temps réel...")

try:
    for index, row in df.iterrows():
        # Extraction des 6 features dans l'ordre exact
        input_data = row[features_cols].values.reshape(1, -1)
        
        # Inférence avec le modèle SVC
        prediction = model.predict(input_data)[0]
        
        # Préparation du message pour le Dashboard Streamlit
        payload = {
            "timestamp_csv": row['Timestamp'],
            "current": float(row['Current_A']),
            "voltage": float(row['Voltage_V']),
            "temp": float(row['Temperature_C']),
            "power": float(row['Power_W']),
            "anomaly_score": int(prediction), # 0 ou 1
            "real_time": time.strftime("%H:%M:%S")
        }
        
        # Envoi MQTT
        client.publish(MQTT_TOPIC, json.dumps(payload))
        
        # Affichage console pour contrôle
        status = "⚠️ FAULT" if prediction == 1 else "✅ OK"
        print(f"Time: {payload['real_time']} | Status: {status} | Power: {payload['power']:.2f}W")
        
        time.sleep(1) # Simule une lecture par seconde

except KeyboardInterrupt:
    client.disconnect()