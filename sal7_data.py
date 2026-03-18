import pandas as pd

# Lire le fichier
df = pd.read_csv('dataset_standardized.csv')

# Convertir le timestamp en nanosecondes
# (ajuster selon la vraie nature de votre timestamp)
start_time = 1700000000000000000  # Timestamp de départ en ns

with open('dataset_influxdb_line_protocol.txt', 'w') as f:
    for idx, row in df.iterrows():
        timestamp = int(start_time + float(row['Timestamp']) * 1e9)
        
        # Ligne de protocol pour toutes les mesures ensemble
        line = (f"sensor_data"
                f",fault_label={int(row['Fault_Label'])}"
                f" Current_A={row['Current_A']},"
                f"Voltage_V={row['Voltage_V']},"
                f"Temperature_C={row['Temperature_C']},"
                f"Power_W={row['Power_W']},"
                f"Delta_V={row['Delta_V']}"
                f" {timestamp}\n")
        
        f.write(line)

print("Fichier converti en format Line Protocol")