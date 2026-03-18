import streamlit as st
import paho.mqtt.client as mqtt
import json
import pandas as pd
import queue
import time
from datetime import datetime

# 1. Configuration de la page
st.set_page_config(page_title="Industrial AI Monitor", layout="wide")

# 2. Gestion des données (File d'attente + Historique)
if 'msg_queue' not in st.session_state:
    st.session_state.msg_queue = queue.Queue()
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 3. CALLBACK MQTT ---
def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        userdata.put(payload)
    except:
        pass

# --- 4. INITIALISATION MQTT ---
@st.cache_resource
def init_mqtt():
    q = queue.Queue()
    client = mqtt.Client(userdata=q)
    client.on_message = on_message
    client.connect("localhost", 1883, 60)
    client.subscribe("factory/machine_1/inference")
    client.loop_start()
    return client, q

mqtt_client, data_queue = init_mqtt()

# Transfert des données vers la session Streamlit
while not data_queue.empty():
    msg = data_queue.get()
    msg['Time'] = datetime.now().strftime("%H:%M:%S")
    st.session_state.history.append(msg)
    if len(st.session_state.history) > 50: # On garde les 50 derniers points
        st.session_state.history.pop(0)

# --- 5. INTERFACE DASHBOARD ---
st.title("🏭 Dashboard de Maintenance Prédictive")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    last = df.iloc[-1]
    
    # --- LOGIQUE DE DÉTECTION ROBUSTE ---
    # On force la conversion en entier et on vérifie plusieurs noms de clés possibles
    # au cas où il y aurait une majuscule ou une faute de frappe
    raw_score = last.get('anomaly_score', last.get('Anomaly', last.get('Fault_Label', 0)))
    score = int(raw_score) 

    # --- AFFICHAGE DE DÉBOGAGE (À supprimer après le test) ---
    # st.write(f"DEBUG - Score reçu : {score}") 

    st.subheader("💓 État de Santé du Moteur")
    
    # La condition doit être explicite
    if score == 1:
        # ÉTAT D'ALERTE
        st.error("🚨 DÉFAILLANCE DÉTECTÉE ! Arrêt machine recommandé.")
        st.progress(100) # Barre pleine et rouge (Streamlit gère la couleur auto avec st.error)
    else:
        # ÉTAT NORMAL
        st.success("✅ FONCTIONNEMENT NOMINAL")
        st.progress(10) # Barre basse et verte
    # --- SECTION 2 : MÉTRIQUES CLÉS ---
    st.write("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Puissance (W)", f"{last['power']:.2f}")
    m2.metric("Courant (A)", f"{last['current']:.2f}")
    m3.metric("Tension (V)", f"{last['voltage']:.2f}")
    m4.metric("Température (°C)", f"{last['temp']:.2f}")

    # --- SECTION 3 : ANALYSE GRAPHIQUE ---
    st.write("---")
    col_v, col_c = st.columns(2)
    
    with col_v:
        st.subheader("⚡ Tension (Voltage_V)")
        st.line_chart(df.set_index('Time')['voltage'], color="#3498db")

    with col_c:
        st.subheader("🔌 Courant (Current_A)")
        st.line_chart(df.set_index('Time')['current'], color="#e67e22")
    
    # Graphique de Puissance en pleine largeur
    st.subheader("🔋 Puissance Totale (Power_W)")
    st.area_chart(df.set_index('Time')['power'], color="#2ecc71")

else:
    st.info("🛰️ En attente de données MQTT (Vérifiez que le script d'inférence tourne)...")

# Rafraîchissement automatique toutes les secondes
time.sleep(1)
st.rerun()